#!/usr/bin/env python3
"""Run focused CPU benchmark items for PyTorch or JAX and emit normalized CSV."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from collections.abc import Callable

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OMP_DYNAMIC",
    "RAYON_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "VECLIB_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "XLA_FLAGS",
)


def runs_from_env() -> tuple[int, int]:
    profile = os.environ.get("PUBLICATION_GATE_PROFILE", "quick").lower()
    default_runs = 15 if profile == "full" else 7
    runs = int(os.environ.get("BENCH_RUNS", default_runs))
    warmups = int(os.environ.get("BENCH_WARMUPS", 3))
    return runs, warmups


def configure_thread_env(num_threads: int) -> None:
    value = str(num_threads)
    xla_multi_thread = "true" if num_threads > 1 else "false"
    os.environ.update(
        {
            "OMP_NUM_THREADS": value,
            "OMP_THREAD_LIMIT": value,
            "OMP_DYNAMIC": "FALSE",
            "RAYON_NUM_THREADS": value,
            "OPENBLAS_NUM_THREADS": value,
            "GOTO_NUM_THREADS": value,
            "MKL_NUM_THREADS": value,
            "VECLIB_MAXIMUM_THREADS": value,
            "VECLIB_NUM_THREADS": value,
            "NUMEXPR_NUM_THREADS": value,
            "BLIS_NUM_THREADS": value,
            "XLA_FLAGS": (
                f"--xla_cpu_multi_thread_eigen={xla_multi_thread} "
                f"intra_op_parallelism_threads={value}"
            ),
        }
    )


def suite_enabled(suite: str) -> bool:
    selected = os.environ.get("PUBLICATION_GATE_SUITE", "all").lower()
    return selected in {"all", suite}


def sizes_for(profile: str, quick: list[int], full: list[int]) -> list[int]:
    return full if profile == "full" else quick


def data(shape: tuple[int, ...], seed: int):
    import numpy as np

    size = int(np.prod(shape))
    values = []
    for i in range(size):
        x = (i * 6364136223846793005 + seed * 1442695040888963407) & ((1 << 64) - 1)
        values.append(((x % 1024) - 512) / 512.0)
    return np.array(values, dtype=np.float64).reshape(shape)


def well_conditioned(n: int, seed: int):
    matrix = data((n, n), seed) * 0.05
    for j in range(n):
        matrix[j, j] += 1.0 + j / max(n, 1)
    return matrix


def spd(n: int, seed: int):
    matrix = well_conditioned(n, seed)
    return matrix.T @ matrix + data_identity(n)


def data_identity(n: int):
    import numpy as np

    return np.eye(n, dtype=np.float64)


def batched_well_conditioned(n: int, batch: int, seed: int):
    import numpy as np

    return np.stack([well_conditioned(n, seed + i) for i in range(batch)], axis=0)


def batched_spd(n: int, batch: int, seed: int):
    import numpy as np

    return np.stack([spd(n, seed + i) for i in range(batch)], axis=0)


def median_iqr(times: list[float]) -> tuple[float, float]:
    values = sorted(times)
    return statistics.median(values), values[(3 * len(values)) // 4] - values[len(values) // 4]


def bench(setup: Callable[[], object], timed: Callable[..., object], sync: Callable[[object], None], runs: int, warmups: int) -> tuple[float, float]:
    for _ in range(warmups):
        inputs = setup()
        sync(timed(*inputs))
    times = []
    for _ in range(runs):
        inputs = setup()  # Input construction stays outside the timed region.
        start = time.perf_counter()
        sync(timed(*inputs))
        times.append((time.perf_counter() - start) * 1000.0)
    return median_iqr(times)


def emit_row(writer: csv.DictWriter, args, suite: str, benchmark: str, shape: str, setup, timed, sync) -> None:
    try:
        median_ms, iqr_ms = bench(setup, timed, sync, args.runs, args.warmups)
        status = "ok"
        median = f"{median_ms:.6f}"
        iqr = f"{iqr_ms:.6f}"
    except Exception as exc:  # Keep table shape stable across partial backend support.
        status = f"error: {exc}"
        median = ""
        iqr = ""
    writer.writerow(
        {
            "suite": suite,
            "benchmark": benchmark,
            "dtype": "f64",
            "threads": args.num_threads,
            "shape": shape,
            "backend": args.backend,
            "median_ms": median,
            "iqr_ms": iqr,
            "status": status,
        }
    )


def run_pytorch(args, writer: csv.DictWriter) -> None:
    import torch

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_threads)

    def tensor(x, requires_grad: bool = False):
        return torch.tensor(x, dtype=torch.float64, requires_grad=requires_grad)

    def sync(value):
        if isinstance(value, torch.Tensor):
            _ = value.numel()

    profile = os.environ.get("PUBLICATION_GATE_PROFILE", "quick").lower()
    if suite_enabled("small"):
        for n in sizes_for(profile, [2, 4, 8], [2, 4, 8, 16, 32]):
            emit_row(
                writer, args, "small", "matmul", f"{n}x{n}",
                lambda n=n: (tensor(data((n, n), 1)), tensor(data((n, n), 2))),
                lambda a, b: a @ b, sync,
            )
            emit_row(
                writer, args, "small", "einsum_ij_jk_ik", f"{n}x{n}",
                lambda n=n: (tensor(data((n, n), 1)), tensor(data((n, n), 2))),
                lambda a, b: torch.einsum("ij,jk->ik", a, b), sync,
            )
            emit_row(
                writer, args, "small", "svd", f"{n}x{n}",
                lambda n=n: (tensor(well_conditioned(n, 3)),),
                lambda a: torch.linalg.svd(a)[1], sync,
            )
            emit_row(
                writer, args, "small", "qr", f"{n}x{n}",
                lambda n=n: (tensor(well_conditioned(n, 4)),),
                lambda a: torch.linalg.qr(a)[0], sync,
            )
            emit_row(
                writer, args, "small", "eigh", f"{n}x{n}",
                lambda n=n: (tensor(spd(n, 5)),),
                lambda a: torch.linalg.eigh(a)[0], sync,
            )
            for rhs_cols in [1, 4]:
                emit_row(
                    writer, args, "small", "solve", f"{n}x{n},rhs={rhs_cols}",
                    lambda n=n, rhs_cols=rhs_cols: (tensor(spd(n, 6)), tensor(data((n, rhs_cols), 7))),
                    lambda a, b: torch.linalg.solve(a, b), sync,
                )
            emit_row(
                writer, args, "small", "grad_sum_matmul_backward", f"{n}x{n}",
                lambda n=n: (tensor(data((n, n), 8), True), tensor(data((n, n), 9), True)),
                lambda a, b: grad_torch_matmul(a, b), sync,
            )
            emit_row(
                writer, args, "small", "grad_sum_svd_s_backward", f"{n}x{n}",
                lambda n=n: (tensor(well_conditioned(n, 10), True),),
                lambda a: grad_torch_svd(a), sync,
            )
            emit_row(
                writer, args, "small", "grad_sum_solve_backward", f"{n}x{n},rhs=1",
                lambda n=n: (tensor(spd(n, 11), True), tensor(data((n, 1), 12), True)),
                lambda a, b: grad_torch_solve(a, b), sync,
            )
    if suite_enabled("large"):
        for n in sizes_for(profile, [128, 256], [128, 256, 512, 1024]):
            emit_row(
                writer, args, "large", "matmul", f"{n}x{n}",
                lambda n=n: (tensor(data((n, n), 21)), tensor(data((n, n), 22))),
                lambda a, b: a @ b, sync,
            )
        rects = [(1024, 256, 1024), (256, 1024, 256)]
        for m, k, n in rects:
            if profile == "quick" and m > 256:
                continue
            emit_row(
                writer, args, "large", "matmul_rect", f"{m}x{k} * {k}x{n}",
                lambda m=m, k=k, n=n: (tensor(data((m, k), 23)), tensor(data((k, n), 24))),
                lambda a, b: a @ b, sync,
            )
        for n in sizes_for(profile, [64], [64, 128, 256]):
            emit_row(
                writer, args, "large", "svd", f"{n}x{n}",
                lambda n=n: (tensor(well_conditioned(n, 25)),),
                lambda a: torch.linalg.svd(a)[1], sync,
            )
            emit_row(
                writer, args, "large", "qr", f"{n}x{n}",
                lambda n=n: (tensor(well_conditioned(n, 26)),),
                lambda a: torch.linalg.qr(a)[0], sync,
            )
            emit_row(
                writer, args, "large", "eigh", f"{n}x{n}",
                lambda n=n: (tensor(spd(n, 27)),),
                lambda a: torch.linalg.eigh(a)[0], sync,
            )
            for rhs_cols in [1, 16, 64]:
                emit_row(
                    writer, args, "large", "solve", f"{n}x{n},rhs={rhs_cols}",
                    lambda n=n, rhs_cols=rhs_cols: (tensor(spd(n, 28)), tensor(data((n, rhs_cols), 29))),
                    lambda a, b: torch.linalg.solve(a, b), sync,
                )
        for n in sizes_for(profile, [64], [64, 128]):
            emit_row(
                writer, args, "large", "grad_sum_matmul", f"{n}x{n}",
                lambda n=n: (tensor(data((n, n), 30), True), tensor(data((n, n), 31), True)),
                lambda a, b: (a @ b).sum(), sync,
            )
            emit_row(
                writer, args, "large", "grad_sum_matmul_backward", f"{n}x{n}",
                lambda n=n: (tensor(data((n, n), 32), True), tensor(data((n, n), 33), True)),
                lambda a, b: grad_torch_matmul(a, b), sync,
            )
            emit_row(
                writer, args, "large", "grad_sum_svd_s_backward", f"{n}x{n}",
                lambda n=n: (tensor(well_conditioned(n, 34), True),),
                lambda a: grad_torch_svd(a), sync,
            )
            emit_row(
                writer, args, "large", "grad_sum_solve_backward", f"{n}x{n},rhs=1",
                lambda n=n: (tensor(spd(n, 35), True), tensor(data((n, 1), 36), True)),
                lambda a, b: grad_torch_solve(a, b), sync,
            )
    if suite_enabled("batched"):
        batches = sizes_for(profile, [16, 64], [16, 64, 256, 1024])
        sizes = sizes_for(profile, [2, 4], [2, 4, 8, 16])
        for batch in batches:
            for n in sizes:
                label = f"{n}x{n}xbatch{batch} (native batch layout)"
                emit_row(
                    writer, args, "batched", "batched_matmul_ikb_kjb_ijb", label,
                    lambda n=n, batch=batch: (tensor(data((batch, n, n), 41)), tensor(data((batch, n, n), 42))),
                    lambda a, b: torch.einsum("bik,bkj->bij", a, b), sync,
                )
                emit_row(
                    writer, args, "batched", "batched_svd", label,
                    lambda n=n, batch=batch: (tensor(batched_well_conditioned(n, batch, 43)),),
                    lambda a: torch.linalg.svd(a)[1], sync,
                )
                emit_row(
                    writer, args, "batched", "batched_qr", label,
                    lambda n=n, batch=batch: (tensor(batched_well_conditioned(n, batch, 44)),),
                    lambda a: torch.linalg.qr(a)[0], sync,
                )
                emit_row(
                    writer, args, "batched", "batched_eigh", label,
                    lambda n=n, batch=batch: (tensor(batched_spd(n, batch, 45)),),
                    lambda a: torch.linalg.eigh(a)[0], sync,
                )
                emit_row(
                    writer, args, "batched", "batched_solve", f"{label},rhs=1",
                    lambda n=n, batch=batch: (tensor(batched_spd(n, batch, 46)), tensor(data((batch, n, 1), 47))),
                    lambda a, b: torch.linalg.solve(a, b), sync,
                )
                emit_row(
                    writer, args, "batched", "grad_sum_batched_matmul_backward", label,
                    lambda n=n, batch=batch: (tensor(data((batch, n, n), 48), True), tensor(data((batch, n, n), 49), True)),
                    lambda a, b: grad_torch_batched_matmul(a, b), sync,
                )
                emit_row(
                    writer, args, "batched", "grad_sum_batched_solve_backward", f"{label},rhs=1",
                    lambda n=n, batch=batch: (tensor(batched_spd(n, batch, 50), True), tensor(data((batch, n, 1), 51), True)),
                    lambda a, b: grad_torch_solve(a, b), sync,
                )


def grad_torch_matmul(a, b):
    loss = (a @ b).sum()
    loss.backward()
    return a.grad


def grad_torch_svd(a):
    loss = torch_svd_s(a).sum()
    loss.backward()
    return a.grad


def torch_svd_s(a):
    import torch

    return torch.linalg.svd(a)[1]


def grad_torch_solve(a, b):
    import torch

    loss = torch.linalg.solve(a, b).sum()
    loss.backward()
    return a.grad


def grad_torch_batched_matmul(a, b):
    import torch

    loss = torch.einsum("bik,bkj->bij", a, b).sum()
    loss.backward()
    return a.grad


def run_jax(args, writer: csv.DictWriter) -> None:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    def array(x):
        return jnp.asarray(x, dtype=jnp.float64)

    def sync(value):
        jax.block_until_ready(value)

    profile = os.environ.get("PUBLICATION_GATE_PROFILE", "quick").lower()
    if suite_enabled("small"):
        for n in sizes_for(profile, [2, 4, 8], [2, 4, 8, 16, 32]):
            emit_row(
                writer, args, "small", "matmul", f"{n}x{n}",
                lambda n=n: (array(data((n, n), 1)), array(data((n, n), 2))),
                lambda a, b: a @ b, sync,
            )
            emit_row(
                writer, args, "small", "einsum_ij_jk_ik", f"{n}x{n}",
                lambda n=n: (array(data((n, n), 1)), array(data((n, n), 2))),
                lambda a, b: jnp.einsum("ij,jk->ik", a, b), sync,
            )
            emit_row(
                writer, args, "small", "svd", f"{n}x{n}",
                lambda n=n: (array(well_conditioned(n, 3)),),
                lambda a: jnp.linalg.svd(a, full_matrices=True)[1], sync,
            )
            emit_row(
                writer, args, "small", "qr", f"{n}x{n}",
                lambda n=n: (array(well_conditioned(n, 4)),),
                lambda a: jnp.linalg.qr(a)[0], sync,
            )
            emit_row(
                writer, args, "small", "eigh", f"{n}x{n}",
                lambda n=n: (array(spd(n, 5)),),
                lambda a: jnp.linalg.eigh(a)[0], sync,
            )
            for rhs_cols in [1, 4]:
                emit_row(
                    writer, args, "small", "solve", f"{n}x{n},rhs={rhs_cols}",
                    lambda n=n, rhs_cols=rhs_cols: (array(spd(n, 6)), array(data((n, rhs_cols), 7))),
                    lambda a, b: jnp.linalg.solve(a, b), sync,
                )
            emit_row(
                writer, args, "small", "grad_sum_matmul_backward", f"{n}x{n}",
                lambda n=n: (array(data((n, n), 8)), array(data((n, n), 9))),
                lambda a, b: jax.grad(lambda x: jnp.sum(x @ b))(a), sync,
            )
            emit_row(
                writer, args, "small", "grad_sum_svd_s_backward", f"{n}x{n}",
                lambda n=n: (array(well_conditioned(n, 10)),),
                lambda a: jax.grad(lambda x: jnp.sum(jnp.linalg.svd(x, full_matrices=True)[1]))(a), sync,
            )
            emit_row(
                writer, args, "small", "grad_sum_solve_backward", f"{n}x{n},rhs=1",
                lambda n=n: (array(spd(n, 11)), array(data((n, 1), 12))),
                lambda a, b: jax.grad(lambda x: jnp.sum(jnp.linalg.solve(x, b)))(a), sync,
            )
    if suite_enabled("large"):
        for n in sizes_for(profile, [128, 256], [128, 256, 512, 1024]):
            emit_row(
                writer, args, "large", "matmul", f"{n}x{n}",
                lambda n=n: (array(data((n, n), 21)), array(data((n, n), 22))),
                lambda a, b: a @ b, sync,
            )
        for m, k, n in [(1024, 256, 1024), (256, 1024, 256)]:
            if profile == "quick" and m > 256:
                continue
            emit_row(
                writer, args, "large", "matmul_rect", f"{m}x{k} * {k}x{n}",
                lambda m=m, k=k, n=n: (array(data((m, k), 23)), array(data((k, n), 24))),
                lambda a, b: a @ b, sync,
            )
        for n in sizes_for(profile, [64], [64, 128, 256]):
            emit_row(
                writer, args, "large", "svd", f"{n}x{n}",
                lambda n=n: (array(well_conditioned(n, 25)),),
                lambda a: jnp.linalg.svd(a, full_matrices=True)[1], sync,
            )
            emit_row(
                writer, args, "large", "qr", f"{n}x{n}",
                lambda n=n: (array(well_conditioned(n, 26)),),
                lambda a: jnp.linalg.qr(a)[0], sync,
            )
            emit_row(
                writer, args, "large", "eigh", f"{n}x{n}",
                lambda n=n: (array(spd(n, 27)),),
                lambda a: jnp.linalg.eigh(a)[0], sync,
            )
            for rhs_cols in [1, 16, 64]:
                emit_row(
                    writer, args, "large", "solve", f"{n}x{n},rhs={rhs_cols}",
                    lambda n=n, rhs_cols=rhs_cols: (array(spd(n, 28)), array(data((n, rhs_cols), 29))),
                    lambda a, b: jnp.linalg.solve(a, b), sync,
                )
        for n in sizes_for(profile, [64], [64, 128]):
            emit_row(
                writer, args, "large", "grad_sum_matmul", f"{n}x{n}",
                lambda n=n: (array(data((n, n), 30)), array(data((n, n), 31))),
                lambda a, b: jnp.sum(a @ b), sync,
            )
            emit_row(
                writer, args, "large", "grad_sum_matmul_backward", f"{n}x{n}",
                lambda n=n: (array(data((n, n), 32)), array(data((n, n), 33))),
                lambda a, b: jax.grad(lambda x: jnp.sum(x @ b))(a), sync,
            )
            emit_row(
                writer, args, "large", "grad_sum_svd_s_backward", f"{n}x{n}",
                lambda n=n: (array(well_conditioned(n, 34)),),
                lambda a: jax.grad(lambda x: jnp.sum(jnp.linalg.svd(x, full_matrices=True)[1]))(a), sync,
            )
            emit_row(
                writer, args, "large", "grad_sum_solve_backward", f"{n}x{n},rhs=1",
                lambda n=n: (array(spd(n, 35)), array(data((n, 1), 36))),
                lambda a, b: jax.grad(lambda x: jnp.sum(jnp.linalg.solve(x, b)))(a), sync,
            )
    if suite_enabled("batched"):
        batches = sizes_for(profile, [16, 64], [16, 64, 256, 1024])
        sizes = sizes_for(profile, [2, 4], [2, 4, 8, 16])
        for batch in batches:
            for n in sizes:
                label = f"{n}x{n}xbatch{batch} (native batch layout)"
                emit_row(
                    writer, args, "batched", "batched_matmul_ikb_kjb_ijb", label,
                    lambda n=n, batch=batch: (array(data((batch, n, n), 41)), array(data((batch, n, n), 42))),
                    lambda a, b: jnp.einsum("bik,bkj->bij", a, b), sync,
                )
                emit_row(
                    writer, args, "batched", "batched_svd", label,
                    lambda n=n, batch=batch: (array(batched_well_conditioned(n, batch, 43)),),
                    lambda a: jnp.linalg.svd(a, full_matrices=True)[1], sync,
                )
                emit_row(
                    writer, args, "batched", "batched_qr", label,
                    lambda n=n, batch=batch: (array(batched_well_conditioned(n, batch, 44)),),
                    lambda a: jnp.linalg.qr(a)[0], sync,
                )
                emit_row(
                    writer, args, "batched", "batched_eigh", label,
                    lambda n=n, batch=batch: (array(batched_spd(n, batch, 45)),),
                    lambda a: jnp.linalg.eigh(a)[0], sync,
                )
                emit_row(
                    writer, args, "batched", "batched_solve", f"{label},rhs=1",
                    lambda n=n, batch=batch: (array(batched_spd(n, batch, 46)), array(data((batch, n, 1), 47))),
                    lambda a, b: jnp.linalg.solve(a, b), sync,
                )
                emit_row(
                    writer, args, "batched", "grad_sum_batched_matmul_backward", label,
                    lambda n=n, batch=batch: (array(data((batch, n, n), 48)), array(data((batch, n, n), 49))),
                    lambda a, b: jax.grad(lambda x: jnp.sum(jnp.einsum("bik,bkj->bij", x, b)))(a), sync,
                )
                emit_row(
                    writer, args, "batched", "grad_sum_batched_solve_backward", f"{label},rhs=1",
                    lambda n=n, batch=batch: (array(batched_spd(n, batch, 50)), array(data((batch, n, 1), 51))),
                    lambda a, b: jax.grad(lambda x: jnp.sum(jnp.linalg.solve(x, b)))(a), sync,
                )


def main() -> None:
    runs, warmups = runs_from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pytorch-cpu", "jax-cpu"], required=True)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=runs)
    parser.add_argument("--warmups", type=int, default=warmups)
    args = parser.parse_args()
    configure_thread_env(args.num_threads)

    fieldnames = ["suite", "benchmark", "dtype", "threads", "shape", "backend", "median_ms", "iqr_ms", "status"]
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        if args.backend == "pytorch-cpu":
            run_pytorch(args, writer)
        else:
            run_jax(args, writer)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
