#!/usr/bin/env python3
"""GPU benchmark runner for pytorch-cuda and jax-cuda backends (Phase 2).

Usage (same interface as the Phase 1 inline smoke):
  python benchmark_gpu_python.py OUTPUT_JSONL DEVICE_ORDINAL PROBLEM_FILTER BACKEND... -- SUITE.yaml...
"""
from __future__ import annotations

import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

_PYTORCH_OPS = {"matmul", "batched_matmul", "einsum", "qr", "solve", "svd", "eigh", "spmv", "spmm"}
_JAX_OPS = {"matmul", "batched_matmul", "einsum", "qr", "solve", "svd", "eigh"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    out = Path(sys.argv[1])
    device_ordinal = int(sys.argv[2])
    problem_filter = sys.argv[3]
    sep = sys.argv.index("--")
    backends = sys.argv[4:sep]
    suites = [Path(p) for p in sys.argv[sep + 1:]]

    root = Path(__file__).resolve().parents[1]
    ts = datetime.now(timezone.utc).isoformat()
    bc = _git_commit(root)
    tc = _git_commit(root / "extern" / "tenferro-rs")

    with out.open("w") as fh:
        for sp in suites:
            suite = yaml.safe_load(sp.read_text())
            for prob in suite["problems"]:
                if problem_filter and prob["id"] != problem_filter:
                    continue
                for backend in backends:
                    rec = _run_one(suite["suite_id"], prob, backend, device_ordinal,
                                   ts=ts, bc=bc, tc=tc)
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _run_one(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    kw = dict(ts=ts, bc=bc, tc=tc)
    op = problem["op"]
    pid = problem["id"]
    if backend not in problem["backend_candidates"]:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="unsupported", reason=f"{backend} not listed for {pid}",
                     path="phase2-runner", **kw)
    if backend == "pytorch-cuda":
        if op not in _PYTORCH_OPS:
            return _stub(suite_id, problem, backend, device_ordinal,
                         status="not_configured", reason=f"pytorch-cuda: op={op} not implemented",
                         path="phase2-runner", **kw)
        return _run_pytorch(suite_id, problem, backend, device_ordinal, **kw)
    if backend == "jax-cuda":
        if op not in _JAX_OPS:
            return _stub(suite_id, problem, backend, device_ordinal,
                         status="not_configured", reason=f"jax-cuda: op={op} not implemented",
                         path="phase2-runner", **kw)
        return _run_jax(suite_id, problem, backend, device_ordinal, **kw)
    return _stub(suite_id, problem, backend, device_ordinal,
                 status="not_configured", reason=f"{backend} not in Python GPU runner",
                 path="phase2-runner", **kw)


# ---------------------------------------------------------------------------
# PyTorch runner
# ---------------------------------------------------------------------------

def _run_pytorch(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    import torch

    kw = dict(ts=ts, bc=bc, tc=tc)
    device = torch.device(f"cuda:{device_ordinal}")
    op = problem["op"]
    run_spec = problem.get("run", {})
    n_warmup = run_spec.get("warmups", 3)
    n_runs = run_spec.get("runs", 7)
    verify_spec = problem.get("verify", {})
    rtol = float(verify_spec.get("rtol", 1e-5))
    atol = float(verify_spec.get("atol", 1e-8))
    path = "phase2-measured-pytorch-cuda"

    try:
        data = _make_data_np(problem)
    except Exception as exc:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="runtime_failed", reason=f"data generation: {exc}", path=path, **kw)

    try:
        gpu_data = _to_torch(data, device)
        fn = _torch_fn(op, gpu_data)

        # Warmup
        for _ in range(n_warmup):
            fn()
            torch.cuda.synchronize(device)

        # Timed runs
        times_ms: list[float] = []
        for _ in range(n_runs):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            result = fn()
            torch.cuda.synchronize(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    except torch.cuda.OutOfMemoryError as exc:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="oom", reason=str(exc)[:500], path=path, **kw)
    except Exception as exc:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="runtime_failed", reason=str(exc)[:500], path=path, **kw)

    # Verification against CPU float64
    try:
        cpu_data = _to_torch_cpu_f64(data)
        cpu_fn = _torch_fn(op, cpu_data)
        cpu_result = cpu_fn()
        ver_status, max_abs, max_rel = _verify_torch(op, result, cpu_result, data, rtol, atol)
    except Exception as exc:
        ver_status, max_abs, max_rel = "skipped", None, None

    if ver_status == "failed":
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="verification_failed",
                     reason=f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
                     path=path, ver_status="failed", max_abs=max_abs, max_rel=max_rel,
                     rtol=rtol, atol=atol, **kw)

    first, med, min_t, p95, iqr = _timing_stats(times_ms)
    props = torch.cuda.get_device_properties(device)
    env = _base_env(device_ordinal, ts, bc, tc)
    env.update({
        "gpu_name": props.name,
        "gpu_uuid": str(props.uuid) if hasattr(props, "uuid") else None,
        "gpu_memory_bytes": props.total_memory,
        "cuda_version": torch.version.cuda,
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        "framework_version": f"torch={torch.__version__}",
    })

    return _ok_record(suite_id, problem, backend, device_ordinal,
                      path=path,
                      warmup_runs=n_warmup, timed_runs=n_runs,
                      first_run_ms=first, median_ms=med, min_ms=min_t, p95_ms=p95, iqr_ms=iqr,
                      compile_time_ms=None,
                      ver_status=ver_status, max_abs=max_abs, max_rel=max_rel,
                      rtol=rtol, atol=atol,
                      env=env)


# ---------------------------------------------------------------------------
# JAX runner
# ---------------------------------------------------------------------------

def _run_jax(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    kw = dict(ts=ts, bc=bc, tc=tc)
    op = problem["op"]
    run_spec = problem.get("run", {})
    n_warmup = run_spec.get("warmups", 3)
    n_runs = run_spec.get("runs", 7)
    verify_spec = problem.get("verify", {})
    rtol = float(verify_spec.get("rtol", 1e-5))
    atol = float(verify_spec.get("atol", 1e-8))
    path = "phase2-measured-jax-cuda"

    cuda_devices = jax.devices("cuda")
    if device_ordinal >= len(cuda_devices):
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="not_configured",
                     reason=f"cuda device {device_ordinal} not available (found {len(cuda_devices)})",
                     path=path, **kw)
    dev = cuda_devices[device_ordinal]

    try:
        data = _make_data_np(problem)
    except Exception as exc:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="runtime_failed", reason=f"data generation: {exc}", path=path, **kw)

    try:
        gpu_data = _to_jax(data, dev)
        fn, fn_jit = _jax_fn(op, gpu_data)

        # Warmup (triggers JIT compilation)
        for _ in range(n_warmup):
            jax.block_until_ready(fn_jit())

        # Timed runs (JIT already compiled)
        times_ms: list[float] = []
        for _ in range(n_runs):
            jax.block_until_ready(jnp.zeros(1, device=dev))
            t0 = time.perf_counter()
            result = jax.block_until_ready(fn_jit())
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    except Exception as exc:
        msg = str(exc)
        status = "oom" if "out of memory" in msg.lower() else "runtime_failed"
        return _stub(suite_id, problem, backend, device_ordinal,
                     status=status, reason=msg[:500], path=path, **kw)

    # Verification against CPU float64
    try:
        import numpy as np
        cpu_data_jax = _to_jax_cpu(data)
        cpu_fn_jit = jax.jit(fn.__class__._unwrapped if hasattr(fn, '_unwrapped') else fn)
        _, cpu_fn_jit = _jax_fn(op, cpu_data_jax)
        cpu_result = jax.block_until_ready(cpu_fn_jit())
        ver_status, max_abs, max_rel = _verify_jax(op, result, cpu_result, data, rtol, atol)
    except Exception:
        ver_status, max_abs, max_rel = "skipped", None, None

    if ver_status == "failed":
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="verification_failed",
                     reason=f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
                     path=path, ver_status="failed", max_abs=max_abs, max_rel=max_rel,
                     rtol=rtol, atol=atol, **kw)

    first, med, min_t, p95, iqr = _timing_stats(times_ms)
    env = _base_env(device_ordinal, ts, bc, tc)
    env.update({
        "gpu_name": dev.device_kind,
        "framework_version": f"jax={jax.__version__}",
    })

    return _ok_record(suite_id, problem, backend, device_ordinal,
                      path=path,
                      warmup_runs=n_warmup, timed_runs=n_runs,
                      first_run_ms=first, median_ms=med, min_ms=min_t, p95_ms=p95, iqr_ms=iqr,
                      compile_time_ms=None,
                      ver_status=ver_status, max_abs=max_abs, max_rel=max_rel,
                      rtol=rtol, atol=atol,
                      env=env)


# ---------------------------------------------------------------------------
# Data generation (numpy-based, framework-agnostic)
# ---------------------------------------------------------------------------

def _make_data_np(problem: dict) -> dict:
    """Return raw numpy arrays + metadata for a problem."""
    import numpy as np

    op = problem["op"]
    dtype_spec = problem.get("dtype", {})
    data_spec = problem.get("data", {})
    generator = data_spec.get("generator", "normal")
    seed = int(data_spec.get("seed", 0))
    rng = np.random.default_rng(seed)
    f64 = np.float64

    def normal(*shape):
        return rng.standard_normal(shape).astype(f64)

    def well_conditioned(m, n):
        A = rng.standard_normal((m, n)).astype(f64)
        Q, _ = np.linalg.qr(A if m >= n else A.T)
        return Q if m >= n else Q.T

    def spd(n):
        A = rng.standard_normal((n, n)).astype(f64)
        return A @ A.T + n * np.eye(n, dtype=f64)

    if op == "matmul":
        p = problem["matmul"]
        m, n, k = p["m"], p["n"], p["k"]
        return {"op": op, "A": normal(m, k), "B": normal(k, n),
                "transpose_a": p.get("transpose_a", False),
                "transpose_b": p.get("transpose_b", False)}

    if op == "batched_matmul":
        p = problem["batched_matmul"]
        b, m, n, k = p["batch"], p["m"], p["n"], p["k"]
        return {"op": op, "A": normal(b, m, k), "B": normal(b, k, n)}

    if op == "einsum":
        p = problem["einsum"]
        shapes = p["shapes_rowmajor"]
        tensors = [rng.standard_normal(s).astype(f64) for s in shapes]
        return {"op": op, "expr": p["format_rowmajor"], "tensors": tensors}

    if op == "qr":
        p = problem["linalg"]
        m, n = p["m"], p["n"]
        A = well_conditioned(m, n) if generator == "well_conditioned" else normal(m, n)
        return {"op": op, "A": A, "full_matrices": p.get("full_matrices", True)}

    if op == "solve":
        p = problem["linalg"]
        n = p["n"]
        rhs = p.get("rhs_cols", 1)
        A = spd(n) if generator == "spd" else normal(n, n) + n * np.eye(n, dtype=f64)
        b = normal(n, rhs) if rhs > 1 else normal(n)
        return {"op": op, "A": A, "b": b}

    if op == "svd":
        p = problem["linalg"]
        m, n = p["m"], p["n"]
        A = well_conditioned(m, n) if generator == "well_conditioned" else normal(m, n)
        return {"op": op, "A": A, "full_matrices": p.get("full_matrices", True)}

    if op == "eigh":
        p = problem["linalg"]
        n = p["n"]
        A = spd(n) if generator == "spd" else (lambda R: R + R.T)(normal(n, n)) + n * np.eye(n, dtype=f64)
        return {"op": op, "A": A}

    if op in ("spmv", "spmm"):
        p = problem["sparse"]
        rows, cols, nnz = p["rows"], p["cols"], p["nnz"]
        rhs_cols = p.get("rhs_cols", 1) if op == "spmm" else None
        # Generate random CSR sparse matrix with given dimensions
        row_idx = rng.integers(0, rows, nnz)
        col_idx = rng.integers(0, cols, nnz)
        values = rng.standard_normal(nnz).astype(f64)
        x = rng.standard_normal(cols).astype(f64) if op == "spmv" else rng.standard_normal((cols, rhs_cols)).astype(f64)
        return {"op": op, "rows": rows, "cols": cols,
                "row_idx": row_idx, "col_idx": col_idx, "values": values, "x": x}

    raise ValueError(f"unsupported op: {op}")


# ---------------------------------------------------------------------------
# Framework conversions
# ---------------------------------------------------------------------------

def _to_torch(data: dict, device) -> dict:
    import torch
    out = dict(data)
    dtype = torch.float64
    for key in ("A", "B", "b", "x"):
        if key in data:
            out[key] = torch.as_tensor(data[key], dtype=dtype, device=device)
    if "tensors" in data:
        out["tensors"] = [torch.as_tensor(t, dtype=dtype, device=device) for t in data["tensors"]]
    if "values" in data:
        rows, cols = data["rows"], data["cols"]
        row_idx = torch.as_tensor(data["row_idx"], dtype=torch.int64)
        col_idx = torch.as_tensor(data["col_idx"], dtype=torch.int64)
        vals = torch.as_tensor(data["values"], dtype=dtype)
        sp = torch.sparse_coo_tensor(torch.stack([row_idx, col_idx]), vals, (rows, cols)).coalesce()
        out["sp"] = sp.to_sparse_csr().to(device)
        out["x"] = torch.as_tensor(data["x"], dtype=dtype, device=device)
    return out


def _to_torch_cpu_f64(data: dict) -> dict:
    import torch
    out = dict(data)
    dtype = torch.float64
    for key in ("A", "B", "b", "x"):
        if key in data:
            out[key] = torch.as_tensor(data[key], dtype=dtype)
    if "tensors" in data:
        out["tensors"] = [torch.as_tensor(t, dtype=dtype) for t in data["tensors"]]
    if "values" in data:
        rows, cols = data["rows"], data["cols"]
        row_idx = torch.as_tensor(data["row_idx"], dtype=torch.int64)
        col_idx = torch.as_tensor(data["col_idx"], dtype=torch.int64)
        vals = torch.as_tensor(data["values"], dtype=dtype)
        sp = torch.sparse_coo_tensor(torch.stack([row_idx, col_idx]), vals, (rows, cols)).coalesce()
        out["sp"] = sp.to_sparse_csr()
        out["x"] = torch.as_tensor(data["x"], dtype=dtype)
    return out


def _to_jax(data: dict, device) -> dict:
    import jax
    import numpy as np
    out = dict(data)
    for key in ("A", "B", "b", "x"):
        if key in data:
            out[key] = jax.device_put(data[key].astype(np.float64), device)
    if "tensors" in data:
        out["tensors"] = [jax.device_put(t.astype(np.float64), device) for t in data["tensors"]]
    return out


def _to_jax_cpu(data: dict) -> dict:
    import jax
    import numpy as np
    out = dict(data)
    cpu_dev = jax.devices("cpu")[0]
    for key in ("A", "B", "b", "x"):
        if key in data:
            out[key] = jax.device_put(data[key].astype(np.float64), cpu_dev)
    if "tensors" in data:
        out["tensors"] = [jax.device_put(t.astype(np.float64), cpu_dev) for t in data["tensors"]]
    return out


# ---------------------------------------------------------------------------
# Op callables
# ---------------------------------------------------------------------------

def _torch_fn(op: str, d: dict):
    import torch

    if op == "matmul":
        A, B = d["A"], d["B"]
        ta, tb = d.get("transpose_a", False), d.get("transpose_b", False)
        a = A.T if ta else A
        b = B.T if tb else B
        return lambda: torch.mm(a, b)
    if op == "batched_matmul":
        return lambda: torch.bmm(d["A"], d["B"])
    if op == "einsum":
        return lambda: torch.einsum(d["expr"], *d["tensors"])
    if op == "qr":
        return lambda: torch.linalg.qr(d["A"], mode="reduced" if not d.get("full_matrices") else "complete")
    if op == "solve":
        return lambda: torch.linalg.solve(d["A"], d["b"])
    if op == "svd":
        fm = d.get("full_matrices", True)
        return lambda: torch.linalg.svd(d["A"], full_matrices=fm)
    if op == "eigh":
        return lambda: torch.linalg.eigh(d["A"])
    if op == "spmv":
        return lambda: torch.mv(d["sp"].to_dense(), d["x"])
    if op == "spmm":
        return lambda: torch.mm(d["sp"].to_dense(), d["x"])
    raise ValueError(f"unsupported op: {op}")


def _jax_fn(op: str, d: dict):
    import jax
    import jax.numpy as jnp

    if op == "matmul":
        A, B = d["A"], d["B"]
        ta, tb = d.get("transpose_a", False), d.get("transpose_b", False)
        a = A.T if ta else A
        b = B.T if tb else B
        fn = lambda: jnp.dot(a, b)
    elif op == "batched_matmul":
        A, B = d["A"], d["B"]
        fn = lambda: jnp.einsum("bik,bkj->bij", A, B)
    elif op == "einsum":
        tensors = d["tensors"]
        fn = lambda: jnp.einsum(d["expr"], *tensors)
    elif op == "qr":
        A = d["A"]
        mode = "reduced" if not d.get("full_matrices") else "complete"
        fn = lambda: jnp.linalg.qr(A, mode=mode)
    elif op == "solve":
        A, b = d["A"], d["b"]
        fn = lambda: jnp.linalg.solve(A, b)
    elif op == "svd":
        A = d["A"]
        fm = d.get("full_matrices", True)
        fn = lambda: jnp.linalg.svd(A, full_matrices=fm)
    elif op == "eigh":
        A = d["A"]
        fn = lambda: jnp.linalg.eigh(A)
    else:
        raise ValueError(f"unsupported op: {op}")

    return fn, jax.jit(fn)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_torch(op, gpu_result, cpu_result, data, rtol, atol):
    import torch
    import numpy as np

    def to_np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().to(torch.float64).numpy()
        return t

    def abs_err(a, b):
        return float(np.max(np.abs(to_np(a).ravel() - to_np(b).ravel())))

    def rel_err(a, b):
        diff = np.abs(to_np(a).ravel() - to_np(b).ravel())
        return float(np.max(diff / np.maximum(np.abs(to_np(b).ravel()), 1e-10)))

    def check(a_abs, a_rel, norm_ref=1.0):
        passed = a_abs <= atol + rtol * max(float(norm_ref), 1.0)
        return ("passed" if passed else "failed"), a_abs, a_rel

    try:
        if op in ("matmul", "batched_matmul", "einsum", "solve"):
            cpu_np = to_np(cpu_result)
            a = abs_err(gpu_result, cpu_np)
            r = rel_err(gpu_result, cpu_np)
            return check(a, r, np.max(np.abs(cpu_np)))

        if op == "qr":
            Q_gpu, R_gpu = gpu_result
            A = to_np(data["A"])
            recon = to_np(Q_gpu) @ to_np(R_gpu)
            a = abs_err(recon, A)
            r = rel_err(recon, A)
            return check(a, r, np.max(np.abs(A)))

        if op == "svd":
            U_gpu, S_gpu, Vh_gpu = gpu_result
            A = to_np(data["A"])
            recon = to_np(U_gpu) @ np.diag(to_np(S_gpu)) @ to_np(Vh_gpu)
            a = abs_err(recon, A)
            r = rel_err(recon, A)
            return check(a, r, np.max(np.abs(A)))

        if op == "eigh":
            L_gpu, V_gpu = gpu_result
            A = to_np(data["A"])
            lhs = A @ to_np(V_gpu)
            rhs = to_np(V_gpu) @ np.diag(to_np(L_gpu))
            a = abs_err(lhs, rhs)
            r = rel_err(lhs, rhs)
            return check(a, r, np.max(np.abs(A)))

        if op in ("spmv", "spmm"):
            cpu_np = to_np(cpu_result)
            a = abs_err(gpu_result, cpu_np)
            r = rel_err(gpu_result, cpu_np)
            return check(a, r, np.max(np.abs(cpu_np)))

    except Exception:
        pass

    return "skipped", None, None


def _verify_jax(op, gpu_result, cpu_result, data, rtol, atol):
    import numpy as np

    def to_np(x):
        return np.array(x, dtype=np.float64)

    def abs_err(a, b):
        return float(np.max(np.abs(to_np(a).ravel() - to_np(b).ravel())))

    def rel_err(a, b):
        diff = np.abs(to_np(a).ravel() - to_np(b).ravel())
        return float(np.max(diff / np.maximum(np.abs(to_np(b).ravel()), 1e-10)))

    def check(a_abs, a_rel, norm_ref=1.0):
        passed = a_abs <= atol + rtol * max(float(norm_ref), 1.0)
        return ("passed" if passed else "failed"), a_abs, a_rel

    try:
        if op in ("matmul", "batched_matmul", "einsum", "solve"):
            cpu_np = to_np(cpu_result)
            a = abs_err(gpu_result, cpu_np)
            r = rel_err(gpu_result, cpu_np)
            return check(a, r, np.max(np.abs(cpu_np)))

        if op == "qr":
            Q_gpu, R_gpu = gpu_result
            A = data["A"].astype(np.float64)
            recon = to_np(Q_gpu) @ to_np(R_gpu)
            a = abs_err(recon, A)
            r = rel_err(recon, A)
            return check(a, r, np.max(np.abs(A)))

        if op == "svd":
            U_gpu, S_gpu, Vh_gpu = gpu_result
            A = data["A"].astype(np.float64)
            recon = to_np(U_gpu) @ np.diag(to_np(S_gpu)) @ to_np(Vh_gpu)
            a = abs_err(recon, A)
            r = rel_err(recon, A)
            return check(a, r, np.max(np.abs(A)))

        if op == "eigh":
            L_gpu, V_gpu = gpu_result
            A = data["A"].astype(np.float64)
            lhs = A @ to_np(V_gpu)
            rhs = to_np(V_gpu) @ np.diag(to_np(L_gpu))
            a = abs_err(lhs, rhs)
            r = rel_err(lhs, rhs)
            return check(a, r, np.max(np.abs(A)))

    except Exception:
        pass

    return "skipped", None, None


# ---------------------------------------------------------------------------
# Timing statistics
# ---------------------------------------------------------------------------

def _timing_stats(times_ms: list[float]):
    n = len(times_ms)
    s = sorted(times_ms)
    first = times_ms[0]
    med = statistics.median(times_ms)
    min_t = s[0]
    p95 = s[min(int(0.95 * n), n - 1)]
    q1 = s[max(int(0.25 * n) - 1, 0)]
    q3 = s[min(int(0.75 * n), n - 1)]
    iqr = q3 - q1
    return first, med, min_t, p95, iqr


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def _base_env(device_ordinal, ts, bc, tc):
    return {
        "hostname": socket.gethostname(),
        "timestamp_utc": ts,
        "os": platform.platform(),
        "gpu_name": None,
        "gpu_uuid": None,
        "gpu_memory_bytes": None,
        "driver_version": None,
        "cuda_version": None,
        "cudnn_version": None,
        "framework_version": None,
        "tenferro_rs_commit": tc,
        "benchmark_repo_commit": bc,
        "env": {
            "CUDA_PATH": os.environ.get("CUDA_PATH"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
    }


def _stub(suite_id, problem, backend, device_ordinal, *, status, reason, path,
          ts, bc, tc, ver_status=None, max_abs=None, max_rel=None, rtol=None, atol=None):
    if ver_status is None:
        ver_status = "skipped" if status in ("unsupported", "not_configured") else "skipped"
    if status == "verification_failed":
        ver_status = "failed"
    env = _base_env(device_ordinal, ts, bc, tc)
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
            "status": ver_status,
            "reference_backend": "cpu_fp64" if status == "verification_failed" else None,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "residual": None,
            "rtol": rtol,
            "atol": atol,
            "reason": reason,
        },
        "environment": env,
        "execution": {
            "device": "cuda",
            "device_ordinal": device_ordinal,
            "execution_path": path,
            "synchronization": "not executed",
            "layout": json.dumps(problem.get("layout", {}), sort_keys=True),
            "dtype": json.dumps(problem.get("dtype", {}), sort_keys=True),
            "notes": None,
            "unsupported_reason": reason,
        },
    }


def _ok_record(suite_id, problem, backend, device_ordinal, *, path,
               warmup_runs, timed_runs, first_run_ms, median_ms, min_ms, p95_ms, iqr_ms,
               compile_time_ms, ver_status, max_abs, max_rel, rtol, atol, env):
    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "problem_id": problem["id"],
        "op": problem["op"],
        "backend": backend,
        "status": "ok",
        "timing": {
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
            "compile_time_ms": compile_time_ms,
            "first_run_ms": first_run_ms,
            "median_ms": median_ms,
            "min_ms": min_ms,
            "p95_ms": p95_ms,
            "iqr_ms": iqr_ms,
            "timing_scope": "steady_state_host_api_plus_device_sync",
        },
        "performance": {
            "tflops": None,
            "effective_bandwidth_gbps": None,
            "peak_memory_bytes": None,
        },
        "verification": {
            "status": ver_status,
            "reference_backend": "cpu_fp64",
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "residual": None,
            "rtol": rtol,
            "atol": atol,
            "reason": None,
        },
        "environment": env,
        "execution": {
            "device": "cuda",
            "device_ordinal": device_ordinal,
            "execution_path": path,
            "synchronization": "torch.cuda.synchronize / jax.block_until_ready",
            "layout": json.dumps(problem.get("layout", {}), sort_keys=True),
            "dtype": json.dumps(problem.get("dtype", {}), sort_keys=True),
            "notes": None,
            "unsupported_reason": None,
        },
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()
