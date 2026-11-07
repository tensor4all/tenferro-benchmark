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

import jax_cache_stats  # noqa: E402
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
                        verify_defaults=suite.get("defaults"),
                    )
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")


def run_one(suite_id, problem, backend, device_ordinal, *, ts, bc, tc, verify_defaults=None):
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
        return run_torch(suite_id, problem, backend, device_ordinal, ts=ts, bc=bc, tc=tc,
                         verify_defaults=verify_defaults)
    if backend == "jax-cuda":
        return run_jax(suite_id, problem, backend, device_ordinal, ts=ts, bc=bc, tc=tc,
                       verify_defaults=verify_defaults)
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


def run_torch(suite_id, problem, backend, device_ordinal, *, ts, bc, tc, verify_defaults=None):
    return run_framework(suite_id, problem, backend, device_ordinal, ts=ts, bc=bc, tc=tc,
                         verify_defaults=verify_defaults)


def run_jax(suite_id, problem, backend, device_ordinal, *, ts, bc, tc, verify_defaults=None):
    return run_framework(suite_id, problem, backend, device_ordinal, ts=ts, bc=bc, tc=tc,
                         verify_defaults=verify_defaults)


def run_framework(suite_id, problem, backend, device_ordinal, *, ts, bc, tc, verify_defaults=None):
    import numpy as np
    from gpu_linalg_ad_contracts import cpu_reference, leaves, verify_arrays
    from benchmark_gpu_public_api import measure
    from benchmark_gpu_python import _timing_stats
    import linalg_ad_verify_modes as verify_modes
    is_jax = backend == "jax-cuda"
    path = "prepared-jax-cuda-jit-linalg-ad" if is_jax else "pytorch-cuda-functional-linalg-ad"
    phase, ad = problem["op"], problem["linalg_ad"]
    n, loss_name, seed = ad["n"], ad["loss"], ad["matrix_seed"]
    warmups = int(os.environ.get("BENCH_WARMUPS", problem.get("run", {}).get("warmups", 3)))
    runs = int(os.environ.get("BENCH_RUNS", problem.get("run", {}).get("runs", 7)))
    rtol, atol = problem.get("verify", {}).get("rtol", 1e-5), problem.get("verify", {}).get("atol", 1e-8)
    verify_only = os.environ.get("GPU_BENCH_VERIFY_ONLY") == "1"
    case_started = time.perf_counter()
    # Which check backs this row is the manifest's decision, not the runner's.
    mode = verify_modes.mode_for(problem, verify_defaults)
    try:
        if warmups < 0 or runs < 1:
            raise ValueError("invalid AD sampling counts")
        matrix = matrix_for_linalg_ad(loss_name, n, seed)
        tangent = data((n, n), tangent_seed(seed))
        rhs = data((n, ad.get("rhs_cols", 1)), ad.get("rhs_seed", 0)) if loss_name == "grad_sum_solve" else None
        if mode == "cpu_reference":
            expected, difference = cpu_reference(LOSS_TORCH[loss_name], matrix, tangent, rhs, phase)
        else:
            # analytic and finite_difference build their reference at check time
            # from the same matrix; measure_only builds none at all.
            expected, difference = None, None
        # Input generation plus the CPU AD reference: everything this case pays
        # before its own clocks start.
        fixture_ms = (time.perf_counter() - case_started) * 1000
        compile_ms = None
        cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR") or None
        cache_before = jax_cache_stats.stats(cache_dir)
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
            arrays = [download(v) for v in leaves(result)]
            if mode == "measure_only":
                return None
            if mode == "cpu_reference":
                max_abs, max_rel, worst = verify_arrays(arrays, expected, rtol, atol)
                return {"max_abs_error": max_abs, "max_rel_error": max_rel,
                        "residual": worst, "reference_backend": "cpu_fp64 AD re-execution",
                        "reason": None, "mode": mode, "diagnostic": difference}
            return verify_modes.check(mode, loss_name, phase, arrays, matrix, tangent, rhs,
                                      rtol, atol)
        probe_started = time.perf_counter()
        probe = runner()
        ready(probe)
        probe_ready = time.perf_counter()
        # Compilation plus the first executed call; every timed sample below
        # still starts after warmups, so the measured region is unchanged.
        first_call_ms = (compile_ms or 0.0) + (probe_ready - probe_started) * 1000
        cache_delta = jax_cache_stats.delta(cache_before, jax_cache_stats.stats(cache_dir))
        # Proving the numbers is not a sampling parameter, and this collector used to
        # record nothing about it. It is the CPU reference plus the elementwise
        # comparison of both downloaded outputs, and it happens outside every clock.
        verify_ms = 0.0
        verify_started = time.perf_counter()
        verification = check(probe)
        verify_ms += (time.perf_counter() - verify_started) * 1000
        if mode == "measure_only":
            # No reference was built, so the transferred bytes come from the
            # row's own outputs rather than from a reference-shaped list.
            output_bytes = sum(v.numel() * v.element_size() for v in leaves(probe))
        else:
            output_bytes = sum(a.nbytes for a in expected)
        del probe
        sampling = None
        if not verify_only:
            sampling = measure(runner, sync, ready, output_bytes, warmups, runs,
                               synchronize_each_operation=True)
            final = runner()
            ready(final)
            if mode != "measure_only":
                verify_started = time.perf_counter()
                after = check(final)
                verify_ms += (time.perf_counter() - verify_started) * 1000
                verification = {
                    **verification,
                    "max_abs_error": max(verification["max_abs_error"], after["max_abs_error"]),
                    "max_rel_error": max(verification["max_rel_error"], after["max_rel_error"]),
                    "residual": max(verification["residual"], after["residual"]),
                }
            del final
        stats = (None,)*5 if verify_only else _timing_stats([v/sampling["operations_per_sample"] for v in sampling["batch_ms"]])
        first, median, minimum, p95, iqr = stats
        rec = ok_record(suite_id, problem, backend, device_ordinal, path=path,
                        n_warmup=warmups, n_runs=0 if verify_only else runs, median=median, iqr=iqr,
                        rtol=rtol, atol=atol, ts=ts, bc=bc, tc=tc)
        if verify_only:
            rec["status"] = "verification_only"
        rec["timing"].update(first_run_ms=first, min_ms=minimum, p95_ms=p95, compile_time_ms=compile_ms,
                             first_call_ms=first_call_ms, verification_time_ms=round(verify_ms, 3))
        if mode == "measure_only":
            rec["verification"] = verify_modes.not_run(problem, verify_defaults)
        else:
            rec["verification"].update(status="passed", mode=mode,
                                       reference_backend=verification["reference_backend"],
                                       max_abs_error=verification["max_abs_error"],
                                       max_rel_error=verification["max_rel_error"],
                                       residual=verification["residual"],
                                       reason=verification["reason"])
        rec["execution"]["notes"] = json.dumps(dict(sampling=sampling, central_difference=difference,
            framework_version=version, verification_mode=mode,
            verification_diagnostic=(verification or {}).get("diagnostic"),
            setup="fixtures/transfers, compilation, initialization and validation outside clocks",
            first_call="compilation plus the first executed call, measured outside every timed sample",
            first_call_ms=first_call_ms, compile_time_ms=compile_ms,
            jax_compilation_cache_dir=cache_dir, **cache_delta,
            jax_compilation_cache_note=("directory growth around the case; positive values are new entries, "
                                        "zero is consistent with a hit but is not proof of one"),
            differentiated_argument="A only; RHS fixed", scope="primal scalar loss plus first-order JVP/VJP",
            device_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"), physical_device_index=os.environ.get("GPU_BENCH_PHYSICAL_DEVICE"),
            verification_only=verify_only,
            overhead={"case_wall_ms": (time.perf_counter() - case_started) * 1000,
                      "fixture_ms": fixture_ms,
                      "verify_ms": verify_ms,
                      "timed_ms": sum(sampling["batch_ms"]) if sampling else None,
                      "scope": "case wall clock outside the timed region: input generation, "
                               "CPU AD reference, transfers, compilation, validation, output destruction"}))
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
