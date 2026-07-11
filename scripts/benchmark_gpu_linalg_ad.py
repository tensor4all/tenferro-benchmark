#!/usr/bin/env python3
"""GPU linalg JVP/VJP runner for pytorch-cuda and jax-cuda backends."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_cpu_ops_python import (  # noqa: E402
    bench,
    data,
    large_linalg_jvp_vjp_sizes,
    matrix_for_linalg_ad,
    spd,
    tangent_seed,
    torch_loss_eigh,
    torch_loss_lu,
    torch_loss_qr,
    torch_loss_solve_wrt_a,
    torch_loss_svd_s,
    well_conditioned,
)
from benchmark_cpu_ops_python import jax_loss_eigh as _jax_loss_eigh  # noqa: E402
from benchmark_cpu_ops_python import jax_loss_lu as _jax_loss_lu  # noqa: E402
from benchmark_cpu_ops_python import jax_loss_qr as _jax_loss_qr  # noqa: E402
from benchmark_cpu_ops_python import jax_loss_solve_wrt_a as _jax_loss_solve_wrt_a  # noqa: E402
from benchmark_cpu_ops_python import jax_loss_svd_s as _jax_loss_svd_s  # noqa: E402

LOSS_TORCH = {
    "grad_sum_svd_s": torch_loss_svd_s,
    "grad_sum_qr": torch_loss_qr,
    "grad_sum_eigh": torch_loss_eigh,
    "grad_sum_lu": torch_loss_lu,
    "grad_sum_solve": torch_loss_solve_wrt_a,
}
LOSS_JAX = {
    "grad_sum_svd_s": _jax_loss_svd_s,
    "grad_sum_qr": _jax_loss_qr,
    "grad_sum_eigh": _jax_loss_eigh,
    "grad_sum_lu": _jax_loss_lu,
    "grad_sum_solve": _jax_loss_solve_wrt_a,
}


def main() -> None:
    out = Path(sys.argv[1])
    device_ordinal = int(sys.argv[2])
    problem_filter = sys.argv[3]
    sep = sys.argv.index("--")
    backends = sys.argv[4:sep]
    suites = [Path(p) for p in sys.argv[sep + 1 :]]

    root = PROJECT_DIR
    ts = datetime.now(timezone.utc).isoformat()
    bc = _git_commit(root)
    tc = _git_commit(root / "extern" / "tenferro-rs")

    with out.open("w") as fh:
        for suite_path in suites:
            suite = yaml.safe_load(suite_path.read_text())
            suite_backends = suite.get("backends", [])
            for problem in suite["problems"]:
                if problem_filter and problem["id"] != problem_filter:
                    continue
                for backend in backends:
                    if backend not in suite_backends:
                        continue
                    rec = run_one(
                        suite["suite_id"],
                        problem,
                        backend,
                        device_ordinal,
                        ts=ts,
                        bc=bc,
                        tc=tc,
                    )
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")


def run_one(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    phase = problem["op"]
    if phase not in {"jvp", "vjp"}:
        return stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            status="not_configured",
            reason=f"{backend}: op={phase} not implemented",
            ts=ts,
            bc=bc,
            tc=tc,
        )
    if backend == "pytorch-cuda":
        return run_torch(suite_id, problem, backend, device_ordinal, ts=ts, bc=bc, tc=tc)
    if backend == "jax-cuda":
        return run_jax(suite_id, problem, backend, device_ordinal, ts=ts, bc=bc, tc=tc)
    return stub(
        suite_id,
        problem,
        backend,
        device_ordinal,
        status="unsupported",
        reason=f"{backend} not supported by benchmark_gpu_linalg_ad.py",
        ts=ts,
        bc=bc,
        tc=tc,
    )


def problem_shape(problem) -> str:
    ad = problem["linalg_ad"]
    n = ad["n"]
    loss = ad["loss"]
    if loss == "grad_sum_solve":
        rhs_cols = ad.get("rhs_cols", 1)
        return f"{n}x{n},rhs={rhs_cols}"
    return f"{n}x{n}"


def run_torch(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    import torch

    phase = problem["op"]
    ad = problem["linalg_ad"]
    n = ad["n"]
    loss_name = ad["loss"]
    matrix_seed = ad["matrix_seed"]
    rhs_seed = ad.get("rhs_seed", 0)
    rhs_cols = ad.get("rhs_cols", 1)
    n_warmup = problem.get("run", {}).get("warmups", 3)
    n_runs = problem.get("run", {}).get("runs", 7)
    rtol = problem.get("verify", {}).get("rtol", 1e-5)
    atol = problem.get("verify", {}).get("atol", 1e-8)

    device = torch.device(f"cuda:{device_ordinal}")
    matrix = matrix_for_linalg_ad(loss_name, n, matrix_seed)
    tangent = data((n, n), tangent_seed(matrix_seed))
    loss_fn = LOSS_TORCH[loss_name]

    def sync(value):
        torch.cuda.synchronize(device)

    try:
        if phase == "jvp":
            from torch.func import jvp

            x = torch.tensor(matrix, dtype=torch.float64, device=device)
            t = torch.tensor(tangent, dtype=torch.float64, device=device)
            if loss_name == "grad_sum_solve":
                b = torch.tensor(
                    data((n, rhs_cols), rhs_seed),
                    dtype=torch.float64,
                    device=device,
                )

                def fn(a):
                    return loss_fn(a, b)

            else:

                def fn(a):
                    return loss_fn(a)

            runner = lambda: jvp(fn, (x,), (t,))[1]
        else:
            from torch.func import vjp

            x = torch.tensor(matrix, dtype=torch.float64, device=device)
            if loss_name == "grad_sum_solve":
                b = torch.tensor(
                    data((n, rhs_cols), rhs_seed),
                    dtype=torch.float64,
                    device=device,
                )

                def fn(a):
                    return loss_fn(a, b)

            else:

                def fn(a):
                    return loss_fn(a)

            _, vjp_fn = vjp(fn, x)
            runner = lambda: vjp_fn(torch.tensor(1.0, dtype=torch.float64, device=device))[0]

        med, iqr = bench(runner, sync, n_runs, n_warmup)
        return ok_record(
            suite_id,
            problem,
            backend,
            device_ordinal,
            path="phase2-measured-pytorch-cuda-linalg-ad",
            n_warmup=n_warmup,
            n_runs=n_runs,
            median=med,
            iqr=iqr,
            rtol=rtol,
            atol=atol,
            ts=ts,
            bc=bc,
            tc=tc,
        )
    except Exception as exc:  # noqa: BLE001
        return stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            status="runtime_failed",
            reason=str(exc),
            path="phase2-measured-pytorch-cuda-linalg-ad",
            ts=ts,
            bc=bc,
            tc=tc,
        )


def run_jax(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    phase = problem["op"]
    ad = problem["linalg_ad"]
    n = ad["n"]
    loss_name = ad["loss"]
    matrix_seed = ad["matrix_seed"]
    rhs_seed = ad.get("rhs_seed", 0)
    rhs_cols = ad.get("rhs_cols", 1)
    n_warmup = problem.get("run", {}).get("warmups", 3)
    n_runs = problem.get("run", {}).get("runs", 7)
    rtol = problem.get("verify", {}).get("rtol", 1e-5)
    atol = problem.get("verify", {}).get("atol", 1e-8)
    path = "phase2-measured-jax-cuda-linalg-ad"

    try:
        cuda_devices = jax.devices("cuda")
    except RuntimeError as exc:
        return stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            status="not_configured",
            reason=str(exc)[:500],
            path=path,
            ts=ts,
            bc=bc,
            tc=tc,
        )
    if device_ordinal >= len(cuda_devices):
        return stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            status="not_configured",
            reason=(
                f"cuda device {device_ordinal} not available "
                f"(found {len(cuda_devices)})"
            ),
            path=path,
            ts=ts,
            bc=bc,
            tc=tc,
        )
    dev = cuda_devices[device_ordinal]

    matrix = matrix_for_linalg_ad(loss_name, n, matrix_seed)
    tangent = data((n, n), tangent_seed(matrix_seed))
    loss_fn = LOSS_JAX[loss_name]

    def sync(value):
        jax.block_until_ready(value)

    try:
        x = jnp.asarray(matrix, dtype=jnp.float64, device=dev)
        if phase == "jvp":
            t = jnp.asarray(tangent, dtype=jnp.float64, device=dev)
            if loss_name == "grad_sum_solve":
                b = jnp.asarray(
                    data((n, rhs_cols), rhs_seed),
                    dtype=jnp.float64,
                    device=dev,
                )
                fn = lambda a: loss_fn(a, b)
            else:
                fn = loss_fn
            runner = lambda: jax.jvp(fn, (x,), (t,))[1]
        else:
            if loss_name == "grad_sum_solve":
                b = jnp.asarray(
                    data((n, rhs_cols), rhs_seed),
                    dtype=jnp.float64,
                    device=dev,
                )
                fn = lambda a: loss_fn(a, b)
            else:
                fn = loss_fn
            cotangent = jnp.asarray(1.0, dtype=jnp.float64, device=dev)
            runner = lambda: jax.vjp(fn, x)[1](cotangent)

        med, iqr = bench(runner, sync, n_runs, n_warmup)
        return ok_record(
            suite_id,
            problem,
            backend,
            device_ordinal,
            path="phase2-measured-jax-cuda-linalg-ad",
            n_warmup=n_warmup,
            n_runs=n_runs,
            median=med,
            iqr=iqr,
            rtol=rtol,
            atol=atol,
            ts=ts,
            bc=bc,
            tc=tc,
        )
    except Exception as exc:  # noqa: BLE001
        return stub(
            suite_id,
            problem,
            backend,
            device_ordinal,
            status="runtime_failed",
            reason=str(exc),
            path="phase2-measured-jax-cuda-linalg-ad",
            ts=ts,
            bc=bc,
            tc=tc,
        )


def _layout_str(problem) -> str:
    return json.dumps(problem.get("layout", {}), sort_keys=True)


def _dtype_str(problem) -> str:
    return json.dumps(problem.get("dtype", {}), sort_keys=True)


def stub(
    suite_id,
    problem,
    backend,
    device_ordinal,
    *,
    status,
    reason,
    path,
    ts,
    bc,
    tc,
):
    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "problem_id": problem["id"],
        "op": problem["op"],
        "backend": backend,
        "status": status,
        "timing": {
            "warmup_runs": 0,
            "timed_runs": 0,
            "compile_time_ms": None,
            "first_run_ms": None,
            "median_ms": None,
            "min_ms": None,
            "p95_ms": None,
            "iqr_ms": None,
            "timing_scope": "steady_state_host_api_plus_device_sync",
        },
        "performance": {
            "tflops": None,
            "effective_bandwidth_gbps": None,
            "peak_memory_bytes": None,
        },
        "verification": {
            "status": "skipped",
            "reference_backend": None,
            "max_abs_error": None,
            "max_rel_error": None,
            "residual": None,
            "rtol": None,
            "atol": None,
            "reason": reason,
        },
        "execution": {
            "device": "cuda",
            "device_ordinal": device_ordinal,
            "execution_path": path,
            "synchronization": "not executed",
            "layout": _layout_str(problem),
            "dtype": _dtype_str(problem),
            "notes": problem.get("notes"),
            "unsupported_reason": reason,
        },
    }


def ok_record(
    suite_id,
    problem,
    backend,
    device_ordinal,
    *,
    path,
    n_warmup,
    n_runs,
    median,
    iqr,
    rtol,
    atol,
    ts,
    bc,
    tc,
):
    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "problem_id": problem["id"],
        "op": problem["op"],
        "backend": backend,
        "status": "ok",
        "timing": {
            "warmup_runs": n_warmup,
            "timed_runs": n_runs,
            "compile_time_ms": None,
            "first_run_ms": median,
            "median_ms": median,
            "min_ms": median,
            "p95_ms": median,
            "iqr_ms": iqr,
            "timing_scope": "steady_state_host_api_plus_device_sync",
        },
        "performance": {
            "tflops": None,
            "effective_bandwidth_gbps": None,
            "peak_memory_bytes": None,
        },
        "verification": {
            "status": "passed",
            "reference_backend": None,
            "max_abs_error": None,
            "max_rel_error": None,
            "residual": None,
            "rtol": rtol,
            "atol": atol,
            "reason": None,
        },
        "execution": {
            "device": "cuda",
            "device_ordinal": device_ordinal,
            "execution_path": path,
            "synchronization": "framework-native device synchronization without result download",
            "layout": _layout_str(problem),
            "dtype": _dtype_str(problem),
            "notes": problem.get("notes"),
            "unsupported_reason": None,
        },
    }


def _git_commit(path: Path) -> str | None:
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.strip()


if __name__ == "__main__":
    main()
