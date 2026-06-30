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


def benchmark_enabled(name: str, base: str | None = None) -> bool:
    selected = os.environ.get("CPU_OPS_BENCHMARK_FILTER", "").strip()
    if not selected:
        return True
    allowed = {item.strip() for item in selected.split(",") if item.strip()}
    return name in allowed or (base is not None and base in allowed)


def sizes_for(profile: str, quick: list[int], full: list[int]) -> list[int]:
    return full if profile == "full" else quick


def large_linalg_jvp_vjp_sizes(profile: str) -> list[int]:
    # O(n^3) linalg AD: 256x256 is single-digit ms; 512x512 reaches tens of ms.
    return sizes_for(profile, [256, 512], [256, 512, 1024])


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


def bench(fn: Callable[[], object], sync: Callable[[object], None], runs: int, warmups: int) -> tuple[float, float]:
    for _ in range(warmups):
        sync(fn())
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        sync(fn())
        times.append((time.perf_counter() - start) * 1000.0)
    return median_iqr(times)


TANGENT_SEED_OFFSET = 1000


def tangent_seed(matrix_seed: int) -> int:
    return matrix_seed + TANGENT_SEED_OFFSET


def matrix_for_linalg_ad(op: str, n: int, seed: int):
    if op in {"grad_sum_eigh", "grad_sum_solve"}:
        return spd(n, seed)
    return well_conditioned(n, seed)


def emit_linalg_jvp_vjp_rows(
    writer: csv.DictWriter,
    args,
    suite: str,
    shape: str,
    n: int,
    matrix_seed: int,
    *,
    op: str,
    loss_torch,
    loss_jax,
    rhs_seed: int | None = None,
    rhs_cols: int = 1,
) -> None:
    if not (
        benchmark_enabled(f"{op}_jvp", op)
        or benchmark_enabled(f"{op}_vjp", op)
    ):
        return

    matrix = matrix_for_linalg_ad(op, n, matrix_seed)
    tangent = data((n, n), tangent_seed(matrix_seed))

    def torch_run_jvp():
        import torch
        from torch.func import jvp

        x = torch.tensor(matrix, dtype=torch.float64)
        t = torch.tensor(tangent, dtype=torch.float64)
        if op == "grad_sum_solve":
            b = torch.tensor(data((n, rhs_cols), rhs_seed), dtype=torch.float64)
            _, out = jvp(lambda a: loss_torch(a, b), (x,), (t,))
        else:
            _, out = jvp(loss_torch, (x,), (t,))
        return out

    def torch_run_vjp():
        import torch
        from torch.func import vjp

        x = torch.tensor(matrix, dtype=torch.float64)
        if op == "grad_sum_solve":
            b = torch.tensor(data((n, rhs_cols), rhs_seed), dtype=torch.float64)
            _, vjp_fn = vjp(lambda a: loss_torch(a, b), x)
        else:
            _, vjp_fn = vjp(loss_torch, x)
        return vjp_fn(torch.tensor(1.0, dtype=torch.float64))[0]

    def jax_run_jvp():
        import jax
        import jax.numpy as jnp

        x = jnp.asarray(matrix, dtype=jnp.float64)
        t = jnp.asarray(tangent, dtype=jnp.float64)
        if op == "grad_sum_solve":
            b = jnp.asarray(data((n, rhs_cols), rhs_seed), dtype=jnp.float64)
            fn = lambda a: loss_jax(a, b)
        else:
            fn = loss_jax
        return jax.jvp(fn, (x,), (t,))[1]

    def jax_run_vjp():
        import jax
        import jax.numpy as jnp

        x = jnp.asarray(matrix, dtype=jnp.float64)
        if op == "grad_sum_solve":
            b = jnp.asarray(data((n, rhs_cols), rhs_seed), dtype=jnp.float64)
            fn = lambda a: loss_jax(a, b)
        else:
            fn = loss_jax
        return jax.vjp(fn, x)[1](jnp.array(1.0))

    if args.backend == "pytorch-cpu":
        import torch

        def sync(value):
            if isinstance(value, torch.Tensor):
                _ = value.numel()

        emit_row(writer, args, suite, op, shape, torch_run_jvp, sync, phase="jvp")
        emit_row(writer, args, suite, op, shape, torch_run_vjp, sync, phase="vjp")
    else:
        import jax

        def sync(value):
            jax.block_until_ready(value)

        emit_row(writer, args, suite, op, shape, jax_run_jvp, sync, phase="jvp")
        emit_row(writer, args, suite, op, shape, jax_run_vjp, sync, phase="vjp")


def emit_linalg_jvp_vjp_suite(
    writer: csv.DictWriter,
    args,
    suite: str,
    shape: str,
    n: int,
    *,
    svd_seed: int,
    qr_seed: int,
    eigh_seed: int,
    lu_seed: int,
    solve_matrix_seed: int,
    solve_rhs_seed: int,
) -> None:
    emit_linalg_jvp_vjp_rows(
        writer, args, suite, shape, n, svd_seed,
        op="grad_sum_svd_s", loss_torch=torch_loss_svd_s, loss_jax=jax_loss_svd_s,
    )
    emit_linalg_jvp_vjp_rows(
        writer, args, suite, shape, n, qr_seed,
        op="grad_sum_qr", loss_torch=torch_loss_qr, loss_jax=jax_loss_qr,
    )
    emit_linalg_jvp_vjp_rows(
        writer, args, suite, shape, n, eigh_seed,
        op="grad_sum_eigh", loss_torch=torch_loss_eigh, loss_jax=jax_loss_eigh,
    )
    emit_linalg_jvp_vjp_rows(
        writer, args, suite, shape, n, lu_seed,
        op="grad_sum_lu", loss_torch=torch_loss_lu, loss_jax=jax_loss_lu,
    )
    emit_linalg_jvp_vjp_rows(
        writer,
        args,
        suite,
        f"{shape},rhs=1" if ",rhs=" not in shape else shape,
        n,
        solve_matrix_seed,
        op="grad_sum_solve",
        loss_torch=torch_loss_solve_wrt_a,
        loss_jax=jax_loss_solve_wrt_a,
        rhs_seed=solve_rhs_seed,
        rhs_cols=1,
    )


def torch_loss_svd_s(a):
    import torch

    return torch.linalg.svd(a)[1].sum()


def torch_loss_qr(a):
    import torch

    q, r = torch.linalg.qr(a)
    return q.sum() + r.sum()


def torch_loss_eigh(a):
    import torch

    return torch.linalg.eigh(a)[0].sum()


def torch_loss_lu(a):
    import torch

    _, l, u = torch.linalg.lu(a)
    return l.sum() + u.sum()


def torch_loss_solve_wrt_a(a, b):
    import torch

    return torch.linalg.solve(a, b).sum()


def jax_loss_svd_s(a):
    import jax.numpy as jnp

    return jnp.sum(jnp.linalg.svd(a, full_matrices=True)[1])


def jax_loss_qr(a):
    import jax.numpy as jnp

    q, r = jnp.linalg.qr(a)
    return jnp.sum(q) + jnp.sum(r)


def jax_loss_eigh(a):
    import jax.numpy as jnp

    return jnp.sum(jnp.linalg.eigh(a)[0])


def jax_loss_lu(a):
    import jax.numpy as jnp
    from jax.scipy.linalg import lu as scipy_lu

    _, l, u = scipy_lu(a)
    return jnp.sum(l) + jnp.sum(u)


def jax_loss_solve_wrt_a(a, b):
    import jax.numpy as jnp

    return jnp.sum(jnp.linalg.solve(a, b))


def emit_row(
    writer: csv.DictWriter,
    args,
    suite: str,
    benchmark: str,
    shape: str,
    fn,
    sync,
    *,
    phase: str = "",
) -> None:
    name = f"{benchmark}_{phase}" if phase else benchmark
    if not benchmark_enabled(name, benchmark):
        return

    try:
        median_ms, iqr_ms = bench(fn, sync, args.runs, args.warmups)
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
            "benchmark": name,
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
            emit_row(writer, args, "small", "matmul", f"{n}x{n}", lambda n=n: tensor(data((n, n), 1)) @ tensor(data((n, n), 2)), sync)
            emit_row(writer, args, "small", "einsum_ij_jk_ik", f"{n}x{n}", lambda n=n: torch.einsum("ij,jk->ik", tensor(data((n, n), 1)), tensor(data((n, n), 2))), sync)
            emit_row(writer, args, "small", "svd", f"{n}x{n}", lambda n=n: torch.linalg.svd(tensor(well_conditioned(n, 3)))[1], sync)
            emit_row(writer, args, "small", "qr", f"{n}x{n}", lambda n=n: torch.linalg.qr(tensor(well_conditioned(n, 4)))[0], sync)
            emit_row(writer, args, "small", "eigh", f"{n}x{n}", lambda n=n: torch.linalg.eigh(tensor(spd(n, 5)))[0], sync)
            for rhs_cols in [1, 4]:
                emit_row(writer, args, "small", "solve", f"{n}x{n},rhs={rhs_cols}", lambda n=n, rhs_cols=rhs_cols: torch.linalg.solve(tensor(spd(n, 6)), tensor(data((n, rhs_cols), 7))), sync)
            emit_row(writer, args, "small", "grad_sum_matmul_backward", f"{n}x{n}", lambda n=n: grad_torch_matmul(tensor(data((n, n), 8), True), tensor(data((n, n), 9), True)), sync)
            emit_row(writer, args, "small", "grad_sum_svd_s_backward", f"{n}x{n}", lambda n=n: grad_torch_svd(tensor(well_conditioned(n, 10), True)), sync)
            emit_row(writer, args, "small", "grad_sum_solve_backward", f"{n}x{n},rhs=1", lambda n=n: grad_torch_solve(tensor(spd(n, 11), True), tensor(data((n, 1), 12), True)), sync)
            emit_linalg_jvp_vjp_suite(
                writer,
                args,
                "small",
                f"{n}x{n}",
                n,
                svd_seed=10,
                qr_seed=4,
                eigh_seed=5,
                lu_seed=52,
                solve_matrix_seed=11,
                solve_rhs_seed=12,
            )
    if suite_enabled("large"):
        for n in sizes_for(profile, [128, 256], [128, 256, 512, 1024]):
            emit_row(writer, args, "large", "matmul", f"{n}x{n}", lambda n=n: tensor(data((n, n), 21)) @ tensor(data((n, n), 22)), sync)
        rects = [(1024, 256, 1024), (256, 1024, 256)]
        for m, k, n in rects:
            if profile == "quick" and m > 256:
                continue
            emit_row(writer, args, "large", "matmul_rect", f"{m}x{k} * {k}x{n}", lambda m=m, k=k, n=n: tensor(data((m, k), 23)) @ tensor(data((k, n), 24)), sync)
        for n in sizes_for(profile, [64], [64, 128, 256]):
            emit_row(writer, args, "large", "svd", f"{n}x{n}", lambda n=n: torch.linalg.svd(tensor(well_conditioned(n, 25)))[1], sync)
            emit_row(writer, args, "large", "qr", f"{n}x{n}", lambda n=n: torch.linalg.qr(tensor(well_conditioned(n, 26)))[0], sync)
            emit_row(writer, args, "large", "eigh", f"{n}x{n}", lambda n=n: torch.linalg.eigh(tensor(spd(n, 27)))[0], sync)
            for rhs_cols in [1, 16, 64]:
                emit_row(writer, args, "large", "solve", f"{n}x{n},rhs={rhs_cols}", lambda n=n, rhs_cols=rhs_cols: torch.linalg.solve(tensor(spd(n, 28)), tensor(data((n, rhs_cols), 29))), sync)
        for n in sizes_for(profile, [64], [64, 128]):
            emit_row(writer, args, "large", "grad_sum_matmul", f"{n}x{n}", lambda n=n: (tensor(data((n, n), 30), True) @ tensor(data((n, n), 31), True)).sum(), sync)
            emit_row(writer, args, "large", "grad_sum_matmul_backward", f"{n}x{n}", lambda n=n: grad_torch_matmul(tensor(data((n, n), 32), True), tensor(data((n, n), 33), True)), sync)
            emit_row(writer, args, "large", "grad_sum_svd_s_backward", f"{n}x{n}", lambda n=n: grad_torch_svd(tensor(well_conditioned(n, 34), True)), sync)
            emit_row(writer, args, "large", "grad_sum_solve_backward", f"{n}x{n},rhs=1", lambda n=n: grad_torch_solve(tensor(spd(n, 35), True), tensor(data((n, 1), 36), True)), sync)
        for n in large_linalg_jvp_vjp_sizes(profile):
            emit_linalg_jvp_vjp_suite(
                writer,
                args,
                "large",
                f"{n}x{n}",
                n,
                svd_seed=34,
                qr_seed=26,
                eigh_seed=27,
                lu_seed=54,
                solve_matrix_seed=35,
                solve_rhs_seed=36,
            )
    if suite_enabled("batched"):
        batches = sizes_for(profile, [16, 64], [16, 64, 256, 1024])
        sizes = sizes_for(profile, [2, 4], [2, 4, 8, 16])
        for batch in batches:
            for n in sizes:
                label = f"{n}x{n}xbatch{batch} (native batch layout)"
                emit_row(writer, args, "batched", "batched_matmul_ikb_kjb_ijb", label, lambda n=n, batch=batch: torch.einsum("bik,bkj->bij", tensor(data((batch, n, n), 41)), tensor(data((batch, n, n), 42))), sync)
                emit_row(writer, args, "batched", "batched_svd", label, lambda n=n, batch=batch: torch.linalg.svd(tensor(batched_well_conditioned(n, batch, 43)))[1], sync)
                emit_row(writer, args, "batched", "batched_qr", label, lambda n=n, batch=batch: torch.linalg.qr(tensor(batched_well_conditioned(n, batch, 44)))[0], sync)
                emit_row(writer, args, "batched", "batched_eigh", label, lambda n=n, batch=batch: torch.linalg.eigh(tensor(batched_spd(n, batch, 45)))[0], sync)
                emit_row(writer, args, "batched", "batched_solve", f"{label},rhs=1", lambda n=n, batch=batch: torch.linalg.solve(tensor(batched_spd(n, batch, 46)), tensor(data((batch, n, 1), 47))), sync)
                emit_row(writer, args, "batched", "grad_sum_batched_matmul_backward", label, lambda n=n, batch=batch: grad_torch_batched_matmul(tensor(data((batch, n, n), 48), True), tensor(data((batch, n, n), 49), True)), sync)
                emit_row(writer, args, "batched", "grad_sum_batched_solve_backward", f"{label},rhs=1", lambda n=n, batch=batch: grad_torch_solve(tensor(batched_spd(n, batch, 50), True), tensor(data((batch, n, 1), 51), True)), sync)


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
            emit_row(writer, args, "small", "matmul", f"{n}x{n}", lambda n=n: array(data((n, n), 1)) @ array(data((n, n), 2)), sync)
            emit_row(writer, args, "small", "einsum_ij_jk_ik", f"{n}x{n}", lambda n=n: jnp.einsum("ij,jk->ik", array(data((n, n), 1)), array(data((n, n), 2))), sync)
            emit_row(writer, args, "small", "svd", f"{n}x{n}", lambda n=n: jnp.linalg.svd(array(well_conditioned(n, 3)), full_matrices=True)[1], sync)
            emit_row(writer, args, "small", "qr", f"{n}x{n}", lambda n=n: jnp.linalg.qr(array(well_conditioned(n, 4)))[0], sync)
            emit_row(writer, args, "small", "eigh", f"{n}x{n}", lambda n=n: jnp.linalg.eigh(array(spd(n, 5)))[0], sync)
            for rhs_cols in [1, 4]:
                emit_row(writer, args, "small", "solve", f"{n}x{n},rhs={rhs_cols}", lambda n=n, rhs_cols=rhs_cols: jnp.linalg.solve(array(spd(n, 6)), array(data((n, rhs_cols), 7))), sync)
            emit_row(writer, args, "small", "grad_sum_matmul_backward", f"{n}x{n}", lambda n=n: jax.grad(lambda x: jnp.sum(x @ array(data((n, n), 9))))(array(data((n, n), 8))), sync)
            emit_row(writer, args, "small", "grad_sum_svd_s_backward", f"{n}x{n}", lambda n=n: jax.grad(lambda x: jnp.sum(jnp.linalg.svd(x, full_matrices=True)[1]))(array(well_conditioned(n, 10))), sync)
            emit_row(writer, args, "small", "grad_sum_solve_backward", f"{n}x{n},rhs=1", lambda n=n: jax.grad(lambda x: jnp.sum(jnp.linalg.solve(x, array(data((n, 1), 12)))))(array(spd(n, 11))), sync)
            emit_linalg_jvp_vjp_suite(
                writer,
                args,
                "small",
                f"{n}x{n}",
                n,
                svd_seed=10,
                qr_seed=4,
                eigh_seed=5,
                lu_seed=52,
                solve_matrix_seed=11,
                solve_rhs_seed=12,
            )
    if suite_enabled("large"):
        for n in sizes_for(profile, [128, 256], [128, 256, 512, 1024]):
            emit_row(writer, args, "large", "matmul", f"{n}x{n}", lambda n=n: array(data((n, n), 21)) @ array(data((n, n), 22)), sync)
        for m, k, n in [(1024, 256, 1024), (256, 1024, 256)]:
            if profile == "quick" and m > 256:
                continue
            emit_row(writer, args, "large", "matmul_rect", f"{m}x{k} * {k}x{n}", lambda m=m, k=k, n=n: array(data((m, k), 23)) @ array(data((k, n), 24)), sync)
        for n in sizes_for(profile, [64], [64, 128, 256]):
            emit_row(writer, args, "large", "svd", f"{n}x{n}", lambda n=n: jnp.linalg.svd(array(well_conditioned(n, 25)), full_matrices=True)[1], sync)
            emit_row(writer, args, "large", "qr", f"{n}x{n}", lambda n=n: jnp.linalg.qr(array(well_conditioned(n, 26)))[0], sync)
            emit_row(writer, args, "large", "eigh", f"{n}x{n}", lambda n=n: jnp.linalg.eigh(array(spd(n, 27)))[0], sync)
            for rhs_cols in [1, 16, 64]:
                emit_row(writer, args, "large", "solve", f"{n}x{n},rhs={rhs_cols}", lambda n=n, rhs_cols=rhs_cols: jnp.linalg.solve(array(spd(n, 28)), array(data((n, rhs_cols), 29))), sync)
        for n in sizes_for(profile, [64], [64, 128]):
            emit_row(writer, args, "large", "grad_sum_matmul", f"{n}x{n}", lambda n=n: jnp.sum(array(data((n, n), 30)) @ array(data((n, n), 31))), sync)
            emit_row(writer, args, "large", "grad_sum_matmul_backward", f"{n}x{n}", lambda n=n: jax.grad(lambda x: jnp.sum(x @ array(data((n, n), 33))))(array(data((n, n), 32))), sync)
            emit_row(writer, args, "large", "grad_sum_svd_s_backward", f"{n}x{n}", lambda n=n: jax.grad(lambda x: jnp.sum(jnp.linalg.svd(x, full_matrices=True)[1]))(array(well_conditioned(n, 34))), sync)
            emit_row(writer, args, "large", "grad_sum_solve_backward", f"{n}x{n},rhs=1", lambda n=n: jax.grad(lambda x: jnp.sum(jnp.linalg.solve(x, array(data((n, 1), 36)))))(array(spd(n, 35))), sync)
        for n in large_linalg_jvp_vjp_sizes(profile):
            emit_linalg_jvp_vjp_suite(
                writer,
                args,
                "large",
                f"{n}x{n}",
                n,
                svd_seed=34,
                qr_seed=26,
                eigh_seed=27,
                lu_seed=54,
                solve_matrix_seed=35,
                solve_rhs_seed=36,
            )
    if suite_enabled("batched"):
        batches = sizes_for(profile, [16, 64], [16, 64, 256, 1024])
        sizes = sizes_for(profile, [2, 4], [2, 4, 8, 16])
        for batch in batches:
            for n in sizes:
                label = f"{n}x{n}xbatch{batch} (native batch layout)"
                emit_row(writer, args, "batched", "batched_matmul_ikb_kjb_ijb", label, lambda n=n, batch=batch: jnp.einsum("bik,bkj->bij", array(data((batch, n, n), 41)), array(data((batch, n, n), 42))), sync)
                emit_row(writer, args, "batched", "batched_svd", label, lambda n=n, batch=batch: jnp.linalg.svd(array(batched_well_conditioned(n, batch, 43)), full_matrices=True)[1], sync)
                emit_row(writer, args, "batched", "batched_qr", label, lambda n=n, batch=batch: jnp.linalg.qr(array(batched_well_conditioned(n, batch, 44)))[0], sync)
                emit_row(writer, args, "batched", "batched_eigh", label, lambda n=n, batch=batch: jnp.linalg.eigh(array(batched_spd(n, batch, 45)))[0], sync)
                emit_row(writer, args, "batched", "batched_solve", f"{label},rhs=1", lambda n=n, batch=batch: jnp.linalg.solve(array(batched_spd(n, batch, 46)), array(data((batch, n, 1), 47))), sync)
                emit_row(writer, args, "batched", "grad_sum_batched_matmul_backward", label, lambda n=n, batch=batch: jax.grad(lambda x: jnp.sum(jnp.einsum("bik,bkj->bij", x, array(data((batch, n, n), 49)))))(array(data((batch, n, n), 48))), sync)
                emit_row(writer, args, "batched", "grad_sum_batched_solve_backward", f"{label},rhs=1", lambda n=n, batch=batch: jax.grad(lambda x: jnp.sum(jnp.linalg.solve(x, array(data((batch, n, 1), 51)))))(array(batched_spd(n, batch, 50))), sync)


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
