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
    return run_framework(suite_id, problem, backend, device_ordinal, ts=ts, bc=bc, tc=tc)


def run_jax(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    return run_framework(suite_id, problem, backend, device_ordinal, ts=ts, bc=bc, tc=tc)


def run_framework(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    import numpy as np
    from gpu_linalg_ad_contracts import cpu_reference, leaves, verify_arrays
    from benchmark_gpu_public_api import measure
    from benchmark_gpu_python import _timing_stats
    is_jax = backend == "jax-cuda"
    path = "prepared-jax-cuda-jit-linalg-ad" if is_jax else "pytorch-cuda-functional-linalg-ad"
    phase, ad = problem["op"], problem["linalg_ad"]
    n, loss_name, seed = ad["n"], ad["loss"], ad["matrix_seed"]
    warmups = int(os.environ.get("BENCH_WARMUPS", problem.get("run", {}).get("warmups", 3)))
    runs = int(os.environ.get("BENCH_RUNS", problem.get("run", {}).get("runs", 7)))
    rtol, atol = problem.get("verify", {}).get("rtol", 1e-5), problem.get("verify", {}).get("atol", 1e-8)
    verify_only = os.environ.get("GPU_BENCH_VERIFY_ONLY") == "1"
    try:
        if warmups < 0 or runs < 1:
            raise ValueError("invalid AD sampling counts")
        matrix = matrix_for_linalg_ad(loss_name, n, seed)
        tangent = data((n, n), tangent_seed(seed))
        rhs = data((n, ad.get("rhs_cols", 1)), ad.get("rhs_seed", 0)) if loss_name == "grad_sum_solve" else None
        expected, difference = cpu_reference(LOSS_TORCH[loss_name], matrix, tangent, rhs, phase)
        compile_ms = None
        if is_jax:
            import jax
            import jax.numpy as jnp
            jax.config.update("jax_enable_x64", True)
            jax.config.update("jax_default_matmul_precision", "highest")
            device = jax.devices("cuda")[device_ordinal]
            x, t = (jax.device_put(np.asarray(v, dtype=np.float64), device) for v in (matrix, tangent))
            b = None if rhs is None else jax.device_put(np.asarray(rhs, dtype=np.float64), device)
            loss = LOSS_JAX[loss_name]
            fn = loss if b is None else lambda a: loss(a, b)
            if phase == "jvp":
                operation = lambda a, seed: jax.jvp(fn, (a,), (seed,))
                arguments = x, t
            else:
                def operation(a, seed):
                    primal, pullback = jax.vjp(fn, a)
                    return primal, pullback(seed)
                arguments = x, jax.device_put(np.asarray(1., dtype=np.float64), device)
            jax.block_until_ready((arguments, b))
            started = time.perf_counter()
            compiled = jax.jit(operation).lower(*arguments).compile()
            compile_ms = (time.perf_counter()-started)*1000
            runner = lambda: compiled(*arguments)
            ready = jax.block_until_ready
            sync = lambda: jax.block_until_ready(arguments)
            def check_placement(value):
                if value.devices() != {device} or value.dtype != jnp.float64:
                    raise ValueError("AD output is not F64 on the selected CUDA device")
            download = np.asarray
            version = "jax=" + jax.__version__
        else:
            import torch
            torch.set_num_threads(1)
            torch.cuda.set_device(device_ordinal)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            device = torch.device(f"cuda:{device_ordinal}")
            x, t = (torch.tensor(v, dtype=torch.float64, device=device) for v in (matrix, tangent))
            b = None if rhs is None else torch.tensor(rhs, dtype=torch.float64, device=device)
            loss = LOSS_TORCH[loss_name]
            fn = loss if b is None else lambda a: loss(a, b)
            if phase == "jvp":
                runner = lambda: torch.func.jvp(fn, (x,), (t,))
            else:
                cotangent = torch.tensor(1., dtype=torch.float64, device=device)
                def runner():
                    primal, pullback = torch.func.vjp(fn, x)
                    return primal, pullback(cotangent)
            sync = lambda: torch.cuda.synchronize(device)
            ready = lambda values: sync()
            def check_placement(value):
                if value.device != device or value.dtype != torch.float64:
                    raise ValueError("AD output is not F64 on the selected CUDA device")
            download = lambda value: value.detach().cpu().numpy()
            version = "torch=" + torch.__version__
        def check(result):
            for value in leaves(result):
                check_placement(value)
            return verify_arrays([download(v) for v in leaves(result)], expected, rtol, atol)
        probe = runner()
        ready(probe)
        verification = check(probe)
        output_bytes = sum(a.nbytes for a in expected)
        del probe
        sampling = None
        if not verify_only:
            sampling = measure(runner, sync, ready, output_bytes, warmups, runs)
            final = runner()
            ready(final)
            after = check(final)
            verification = tuple(max(a, b) for a, b in zip(verification, after))
            del final
        stats = (None,)*5 if verify_only else _timing_stats([v/sampling["operations_per_sample"] for v in sampling["batch_ms"]])
        first, median, minimum, p95, iqr = stats
        rec = ok_record(suite_id, problem, backend, device_ordinal, path=path,
                        n_warmup=warmups, n_runs=0 if verify_only else runs, median=median, iqr=iqr,
                        rtol=rtol, atol=atol, ts=ts, bc=bc, tc=tc)
        if verify_only:
            rec["status"] = "verification_only"
        rec["timing"].update(first_run_ms=first, min_ms=minimum, p95_ms=p95, compile_time_ms=compile_ms)
        rec["verification"].update(reference_backend="pytorch-cpu-f64 AD plus central difference of primal loss",
                                   max_abs_error=verification[0], max_rel_error=verification[1], residual=verification[2])
        rec["execution"]["notes"] = json.dumps(dict(sampling=sampling, central_difference=difference,
            framework_version=version, setup="fixtures/transfers, compilation, initialization and validation outside clocks",
            differentiated_argument="A only; RHS fixed", scope="primal scalar loss plus first-order JVP/VJP",
            device_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"), physical_device_index=os.environ.get("GPU_BENCH_PHYSICAL_DEVICE"),
            verification_only=verify_only))
        return rec
    except Exception as exc:
        return stub(suite_id, problem, backend, device_ordinal, status="runtime_failed", reason=str(exc), path=path, ts=ts, bc=bc, tc=tc)


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
