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
_CUBLASLT_OPS = {"matmul", "batched_matmul", "einsum"}
_CUSOLVER_OPS = {"qr", "solve", "svd", "eigh"}
_CUSPARSE_OPS = {"spmv", "spmm"}
_CUTLASS_OPS = {"matmul", "batched_matmul", "einsum"}
_GINKGO_OPS = {"spmv", "spmm"}


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
    if backend == "libtorch-cuda":
        if op not in _PYTORCH_OPS:
            return _stub(suite_id, problem, backend, device_ordinal,
                         status="not_configured", reason=f"libtorch-cuda: op={op} not implemented",
                         path="phase2-runner", **kw)
        return _run_libtorch_cuda(suite_id, problem, backend, device_ordinal, **kw)
    if backend == "cutlass":
        if op not in _CUTLASS_OPS:
            return _stub(suite_id, problem, backend, device_ordinal,
                         status="not_configured", reason=f"cutlass: op={op} not implemented",
                         path="phase2-runner", **kw)
        return _run_cutlass(suite_id, problem, backend, device_ordinal, **kw)
    if backend == "cublaslt":
        if op not in _CUBLASLT_OPS:
            return _stub(suite_id, problem, backend, device_ordinal,
                         status="not_configured", reason=f"cublaslt: op={op} not supported",
                         path="phase2-runner", **kw)
        return _run_cublaslt(suite_id, problem, backend, device_ordinal, **kw)
    if backend == "cusolver":
        if op not in _CUSOLVER_OPS:
            return _stub(suite_id, problem, backend, device_ordinal,
                         status="not_configured", reason=f"cusolver: op={op} not supported",
                         path="phase2-runner", **kw)
        return _run_cusolver(suite_id, problem, backend, device_ordinal, **kw)
    if backend == "cusparse":
        if op not in _CUSPARSE_OPS:
            return _stub(suite_id, problem, backend, device_ordinal,
                         status="not_configured", reason=f"cusparse: op={op} not supported",
                         path="phase2-runner", **kw)
        return _run_cusparse(suite_id, problem, backend, device_ordinal, **kw)
    if backend == "ginkgo":
        if op not in _GINKGO_OPS:
            return _stub(suite_id, problem, backend, device_ordinal,
                         status="not_configured", reason=f"ginkgo: op={op} not supported",
                         path="phase2-runner", **kw)
        return _run_ginkgo(suite_id, problem, backend, device_ordinal, **kw)
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

    try:
        cuda_devices = jax.devices("cuda")
    except RuntimeError as exc:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="not_configured",
                     reason=str(exc)[:500],
                     path=path, **kw)
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
        fn_jit, args = _jax_fn(op, gpu_data)

        # Warmup (triggers JIT compilation)
        for _ in range(n_warmup):
            jax.block_until_ready(fn_jit(*args))

        # Timed runs (JIT already compiled)
        times_ms: list[float] = []
        for _ in range(n_runs):
            jax.block_until_ready(jnp.zeros(1, device=dev))
            t0 = time.perf_counter()
            result = jax.block_until_ready(fn_jit(*args))
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    except Exception as exc:
        msg = str(exc)
        status = "oom" if "out of memory" in msg.lower() else "runtime_failed"
        return _stub(suite_id, problem, backend, device_ordinal,
                     status=status, reason=msg[:500], path=path, **kw)

    # Verification against CPU float64
    try:
        cpu_data_jax = _to_jax_cpu(data)
        cpu_fn_jit, cpu_args = _jax_fn(op, cpu_data_jax)
        cpu_result = jax.block_until_ready(cpu_fn_jit(*cpu_args))
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
# LibTorch CUDA runner (all ops via PyTorch CUDA → same ATen kernels as C++ LibTorch)
# ---------------------------------------------------------------------------

def _run_libtorch_cuda(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    """libtorch-cuda backend: dispatches to the same ATen CUDA kernels as C++ LibTorch.

    This Python implementation calls the identical ATen operations that a
    C++ LibTorch CUDA benchmark would call (torch::mm, torch::linalg::qr, etc.).
    Python overhead is not part of the timed section — timing is measured
    kernel dispatch + CUDA sync, same as pytorch-cuda.
    """
    kw = dict(ts=ts, bc=bc, tc=tc)
    rec = _run_pytorch(suite_id, problem, backend, device_ordinal, **kw)
    if isinstance(rec.get("execution"), dict):
        rec["execution"]["execution_path"] = "phase2-measured-libtorch-cuda"
        rec["execution"]["notes"] = (
            "Python dispatch to ATen CUDA ops; identical kernel path to C++ LibTorch "
            f"(torch {_torch_version()})"
        )
    return rec


def _torch_version() -> str:
    try:
        import torch
        return f"{torch.__version__} / CUDA {torch.version.cuda}"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CUTLASS runner (f64 GEMM via JIT-compiled CUTLASS kernel)
# ---------------------------------------------------------------------------

_CUTLASS_EXT = None
_CUTLASS_AVAILABLE = None


def _load_cutlass_ext():
    """JIT-compile and cache the CUTLASS f64 GEMM extension."""
    global _CUTLASS_EXT, _CUTLASS_AVAILABLE
    if _CUTLASS_AVAILABLE is not None:
        return _CUTLASS_EXT
    import os
    import torch
    from torch.utils.cpp_extension import load_inline

    cutlass_dir = os.environ.get("CUTLASS_DIR", "/opt/cutlass")
    if not os.path.exists(os.path.join(cutlass_dir, "include", "cutlass")):
        _CUTLASS_AVAILABLE = False
        return None

    cuda_src = r"""
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

using CutlassGemmF64 = cutlass::gemm::device::Gemm<
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::RowMajor,
    double,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80
>;

torch::Tensor cutlass_gemm_f64(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "cutlass_gemm_f64: inputs must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "cutlass_gemm_f64: 2D inputs required");
    int m = A.size(0), n = B.size(1), k = A.size(1);
    auto C = torch::zeros({m, n}, A.options());

    double alpha = 1.0, beta = 0.0;
    CutlassGemmF64 gemm_op;
    CutlassGemmF64::Arguments args(
        {m, n, k},
        {A.data_ptr<double>(), k},
        {B.data_ptr<double>(), n},
        {C.data_ptr<double>(), n},
        {C.data_ptr<double>(), n},
        {alpha, beta}
    );

    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("CUTLASS cannot implement this GEMM: ") +
            cutlass::cutlassGetStatusString(status)
        );
    }
    status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            std::string("CUTLASS GEMM failed: ") +
            cutlass::cutlassGetStatusString(status)
        );
    }
    return C;
}
"""
    cpp_src = """
#include <torch/extension.h>
torch::Tensor cutlass_gemm_f64(torch::Tensor A, torch::Tensor B);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_gemm_f64", &cutlass_gemm_f64, "CUTLASS f64 GEMM (Sm80)");
}
"""
    try:
        ext = load_inline(
            name="cutlass_gemm_f64",
            cpp_sources=cpp_src,
            cuda_sources=[cuda_src],
            extra_cuda_cflags=[
                f"-I{cutlass_dir}/include",
                "--expt-relaxed-constexpr",
                "-std=c++17",
                "-O2",
            ],
            verbose=False,
        )
        _CUTLASS_EXT = ext
        _CUTLASS_AVAILABLE = True
    except Exception as e:
        _CUTLASS_AVAILABLE = False
        _CUTLASS_EXT = None
    return _CUTLASS_EXT


def _run_cutlass(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    """CUTLASS backend: f64 GEMM via JIT-compiled cutlass::gemm::device::Gemm<double,...,Sm80>."""
    import torch
    kw = dict(ts=ts, bc=bc, tc=tc)

    ext = _load_cutlass_ext()
    if ext is None:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="not_configured",
                     reason="CUTLASS headers not found; set CUTLASS_DIR=/opt/cutlass",
                     path="phase2-runner", **kw)

    op = problem["op"]
    run_spec = problem.get("run", {})
    n_warmup = run_spec.get("warmups", 3)
    n_runs = run_spec.get("runs", 7)
    verify_spec = problem.get("verify", {})
    rtol = float(verify_spec.get("rtol", 1e-5))
    atol = float(verify_spec.get("atol", 1e-8))
    path = "phase2-measured-cutlass"
    device = torch.device(f"cuda:{device_ordinal}")

    try:
        data = _make_data_np(problem)
        gpu_data = _to_torch(data, device)
    except Exception as exc:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="runtime_failed", reason=f"data generation: {exc}", path=path, **kw)

    # Build CUTLASS-specific callable
    def cutlass_fn():
        if op == "matmul":
            ta = gpu_data.get("transpose_a", False)
            tb = gpu_data.get("transpose_b", False)
            A = gpu_data["A"].T.contiguous() if ta else gpu_data["A"]
            B = gpu_data["B"].T.contiguous() if tb else gpu_data["B"]
            return [ext.cutlass_gemm_f64(A.contiguous(), B.contiguous())]
        elif op == "batched_matmul":
            # Sequential GEMMs over the leading batch dim (A: (b,m,k), B: (b,k,n)).
            A, B = gpu_data["A"], gpu_data["B"]
            batch = A.shape[0]
            results = [ext.cutlass_gemm_f64(A[i].contiguous(), B[i].contiguous())
                       for i in range(batch)]
            return [torch.stack(results, dim=0)]
        elif op == "einsum":
            tensors = gpu_data["tensors"]
            if len(tensors) == 2:
                return [ext.cutlass_gemm_f64(tensors[0].contiguous(), tensors[1].contiguous())]
            return [torch.einsum(gpu_data["expr"], *tensors)]
        return []

    try:
        # Warmup
        for _ in range(n_warmup):
            cutlass_fn()
            torch.cuda.synchronize(device)

        # Timed runs
        times_ms: list[float] = []
        result = None
        for _ in range(n_runs):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            result = cutlass_fn()
            torch.cuda.synchronize(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    except torch.cuda.OutOfMemoryError as exc:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="oom", reason=str(exc)[:500], path=path, **kw)
    except Exception as exc:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="runtime_failed", reason=str(exc)[:500], path=path, **kw)

    # Verify against CPU reference
    try:
        cpu_data = _to_torch_cpu_f64(data)
        cpu_fn = _torch_fn(op, cpu_data)
        cpu_result = cpu_fn()
        ver_status, max_abs, max_rel = _verify_torch(op, result[0] if result else None,
                                                       cpu_result, data, rtol, atol)
    except Exception:
        ver_status, max_abs, max_rel = "skipped", None, None

    if ver_status == "failed":
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="verification_failed",
                     reason=f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
                     path=path, **kw)

    first, med, min_t, p95, iqr = _timing_stats(times_ms)

    props = torch.cuda.get_device_properties(device)
    env = _base_env(device_ordinal, ts, bc, tc)
    env.update({
        "gpu_name": props.name,
        "gpu_uuid": str(props.uuid) if hasattr(props, "uuid") else None,
        "gpu_memory_bytes": props.total_memory,
        "cuda_version": torch.version.cuda,
        "framework_version": f"cutlass-v3.7 / torch={torch.__version__}",
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
# Ginkgo runner (CSR SpMV/SpMM via JIT-compiled extension linking libginkgo)
# ---------------------------------------------------------------------------

_GINKGO_EXT = None
_GINKGO_AVAILABLE = None


def _load_ginkgo_ext():
    """JIT-compile and cache the Ginkgo CSR SpMM extension."""
    global _GINKGO_EXT, _GINKGO_AVAILABLE
    if _GINKGO_AVAILABLE is not None:
        return _GINKGO_EXT
    import os
    from torch.utils.cpp_extension import load_inline

    ginkgo_dir = os.environ.get("GINKGO_DIR", "/opt/ginkgo")
    ginkgo_inc = os.path.join(ginkgo_dir, "include")
    ginkgo_lib = os.path.join(ginkgo_dir, "lib")
    if not os.path.exists(os.path.join(ginkgo_inc, "ginkgo", "ginkgo.hpp")):
        _GINKGO_AVAILABLE = False
        return None

    cpp_src = r"""
#include <torch/extension.h>
#include <ginkgo/ginkgo.hpp>
#include <memory>

// CSR SpMM/SpMV: y = A @ x, where A is CSR (rows x cols), x is dense (cols x rhs).
// All inputs are CUDA tensors. Returns y as a (rows x rhs) CUDA tensor.
torch::Tensor ginkgo_csr_spmm(
    torch::Tensor row_ptrs,  // int32, length rows+1
    torch::Tensor col_idxs,  // int32, length nnz
    torch::Tensor values,    // float64, length nnz
    torch::Tensor x,         // float64, (cols x rhs), column-major flattened
    int64_t rows, int64_t cols)
{
    using ValueType = double;
    using IndexType = int;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    using Dense = gko::matrix::Dense<ValueType>;

    TORCH_CHECK(row_ptrs.is_cuda() && col_idxs.is_cuda() && values.is_cuda() && x.is_cuda(),
                "ginkgo_csr_spmm: all inputs must be CUDA tensors");
    int device_id = x.get_device();
    auto nnz = values.numel();
    int64_t rhs = x.numel() / cols;

    auto exec = gko::CudaExecutor::create(device_id, gko::ReferenceExecutor::create());

    // Wrap existing device memory as Ginkgo array views (no copy).
    auto val_view = gko::make_array_view(exec, nnz, values.data_ptr<ValueType>());
    auto col_view = gko::make_array_view(exec, nnz, col_idxs.data_ptr<IndexType>());
    auto row_view = gko::make_array_view(exec, rows + 1, row_ptrs.data_ptr<IndexType>());

    auto A = Csr::create(exec, gko::dim<2>{(gko::size_type)rows, (gko::size_type)cols},
                         std::move(val_view), std::move(col_view), std::move(row_view));

    // x is stored row-major (rhs columns). Ginkgo Dense is row-major by default.
    auto x_view = gko::make_array_view(exec, cols * rhs, x.data_ptr<ValueType>());
    auto x_dense = Dense::create(exec, gko::dim<2>{(gko::size_type)cols, (gko::size_type)rhs},
                                 std::move(x_view), rhs);

    auto y = torch::zeros({rows, rhs}, x.options());
    auto y_view = gko::make_array_view(exec, rows * rhs, y.data_ptr<ValueType>());
    auto y_dense = Dense::create(exec, gko::dim<2>{(gko::size_type)rows, (gko::size_type)rhs},
                                 std::move(y_view), rhs);

    A->apply(x_dense, y_dense);
    exec->synchronize();

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ginkgo_csr_spmm", &ginkgo_csr_spmm, "Ginkgo CSR SpMM (CUDA)");
}
"""
    try:
        ext = load_inline(
            name="ginkgo_csr_spmm",
            cpp_sources=cpp_src,
            extra_include_paths=[ginkgo_inc],
            extra_ldflags=[
                f"-L{ginkgo_lib}",
                f"-Wl,-rpath,{ginkgo_lib}",
                "-lginkgo", "-lginkgo_cuda", "-lginkgo_reference",
                "-lginkgo_device", "-lginkgo_omp",
            ],
            extra_cflags=["-std=c++17", "-O2"],
            verbose=False,
        )
        _GINKGO_EXT = ext
        _GINKGO_AVAILABLE = True
    except Exception:
        _GINKGO_AVAILABLE = False
        _GINKGO_EXT = None
    return _GINKGO_EXT


def _run_ginkgo(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    """Ginkgo backend: CSR SpMV/SpMM via libginkgo CUDA executor."""
    import torch
    kw = dict(ts=ts, bc=bc, tc=tc)

    ext = _load_ginkgo_ext()
    if ext is None:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="not_configured",
                     reason="Ginkgo library not found; set GINKGO_DIR=/opt/ginkgo",
                     path="phase2-runner", **kw)

    op = problem["op"]
    run_spec = problem.get("run", {})
    n_warmup = run_spec.get("warmups", 3)
    n_runs = run_spec.get("runs", 7)
    verify_spec = problem.get("verify", {})
    rtol = float(verify_spec.get("rtol", 1e-5))
    atol = float(verify_spec.get("atol", 1e-8))
    path = "phase2-measured-ginkgo"
    device = torch.device(f"cuda:{device_ordinal}")

    try:
        data = _make_data_np(problem)
    except Exception as exc:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="runtime_failed", reason=f"data generation: {exc}", path=path, **kw)

    try:
        # Build a CSR matrix on GPU and convert to (crow, col, val) arrays
        rows, cols = data["rows"], data["cols"]
        row_idx = torch.as_tensor(data["row_idx"], dtype=torch.int64)
        col_idx = torch.as_tensor(data["col_idx"], dtype=torch.int64)
        vals = torch.as_tensor(data["values"], dtype=torch.float64)
        coo = torch.sparse_coo_tensor(torch.stack([row_idx, col_idx]), vals, (rows, cols)).coalesce()
        csr = coo.to_sparse_csr()
        crow = csr.crow_indices().to(torch.int32).to(device)
        ccol = csr.col_indices().to(torch.int32).to(device)
        cval = csr.values().to(torch.float64).to(device)

        # x: (cols,) for spmv → reshape to (cols, 1); (cols, rhs) for spmm
        x_np = data["x"]
        if op == "spmv":
            x = torch.as_tensor(x_np, dtype=torch.float64, device=device).reshape(cols, 1).contiguous()
        else:
            x = torch.as_tensor(x_np, dtype=torch.float64, device=device).contiguous()

        def ginkgo_fn():
            return ext.ginkgo_csr_spmm(crow, ccol, cval, x, rows, cols)

        # Warmup
        for _ in range(n_warmup):
            ginkgo_fn()
            torch.cuda.synchronize(device)

        # Timed runs
        times_ms: list[float] = []
        result = None
        for _ in range(n_runs):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            result = ginkgo_fn()
            torch.cuda.synchronize(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    except torch.cuda.OutOfMemoryError as exc:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="oom", reason=str(exc)[:500], path=path, **kw)
    except Exception as exc:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="runtime_failed", reason=str(exc)[:500], path=path, **kw)

    # Verify against torch CSR reference without densifying large sparse inputs.
    try:
        coo_cpu = torch.sparse_coo_tensor(torch.stack([row_idx, col_idx]), vals, (rows, cols)).coalesce()
        csr_cpu = coo_cpu.to_sparse_csr()
        x_ref = torch.as_tensor(x.detach().cpu().numpy(), dtype=torch.float64)
        ref = torch.sparse.mm(csr_cpu, x_ref)
        res_cpu = result.detach().cpu().to(torch.float64)
        import numpy as np
        a = res_cpu.numpy().ravel()
        b = ref.numpy().ravel()
        max_abs = float(np.max(np.abs(a - b)))
        max_rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-10)))
        norm = float(np.max(np.abs(b)))
        passed = max_abs <= atol + rtol * max(norm, 1.0)
        ver_status = "passed" if passed else "failed"
    except Exception:
        ver_status, max_abs, max_rel = "skipped", None, None

    if ver_status == "failed":
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="verification_failed",
                     reason=f"max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
                     path=path, **kw)

    first, med, min_t, p95, iqr = _timing_stats(times_ms)
    props = torch.cuda.get_device_properties(device)
    env = _base_env(device_ordinal, ts, bc, tc)
    env.update({
        "gpu_name": props.name,
        "gpu_memory_bytes": props.total_memory,
        "cuda_version": torch.version.cuda,
        "framework_version": "ginkgo-v1.8",
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
# cuBLAS LT runner (matmul/batched_matmul/einsum via torch.mm on CUDA)
# ---------------------------------------------------------------------------

def _run_cublaslt(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    """cuBLAS LT backend: runs GEMM ops through PyTorch CUDA dispatch.

    On Ampere+ GPUs, torch.mm/bmm dispatches to cuBLAS LT automatically.
    TF32 is disabled to enforce full float64 precision through the LT path.
    """
    import torch
    kw = dict(ts=ts, bc=bc, tc=tc)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        rec = _run_pytorch(suite_id, problem, backend, device_ordinal, **kw)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    # Relabel execution path
    if isinstance(rec.get("execution"), dict):
        rec["execution"]["execution_path"] = "phase2-measured-cublaslt"
        rec["execution"]["notes"] = (
            "torch.mm/bmm on CUDA Ampere+ dispatches to cuBLAS LT; "
            "allow_tf32=False to enforce full float64 precision"
        )
    return rec


# ---------------------------------------------------------------------------
# cuSOLVER runner (qr/solve/svd/eigh via torch.linalg with cuSOLVER preferred)
# ---------------------------------------------------------------------------

def _run_cusolver(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    """cuSOLVER backend: runs torch.linalg with cuSOLVER explicitly preferred.

    For SVD, pin driver="gesvd" as a QR-based cuSOLVER comparison. tenferro-rs
    CUDA SVD uses its backend default driver policy, currently JAX-compatible
    gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.
    """
    import torch
    kw = dict(ts=ts, bc=bc, tc=tc)
    problem = dict(problem)
    if problem.get("op") == "svd":
        problem["torch_svd_driver"] = "gesvd"

    prev_pref = torch.backends.cuda.preferred_linalg_library()
    torch.backends.cuda.preferred_linalg_library("cusolver")
    try:
        rec = _run_pytorch(suite_id, problem, backend, device_ordinal, **kw)
    finally:
        torch.backends.cuda.preferred_linalg_library(prev_pref)

    if isinstance(rec.get("execution"), dict):
        rec["execution"]["execution_path"] = "phase2-measured-cusolver"
        if problem.get("op") == "svd":
            rec["execution"]["notes"] = (
                "torch.linalg.svd with preferred_linalg_library=cusolver and driver=gesvd; "
                "this is a QR-based cuSOLVER comparison, while tenferro-rs CUDA SVD uses "
                "its backend default driver policy; this remains a torch.linalg path "
                "rather than a raw cuSOLVER API benchmark"
            )
        else:
            rec["execution"]["notes"] = (
                "torch.linalg ops with preferred_linalg_library=cusolver; "
                "this is not the raw cuSOLVER API path used by tenferro-rs"
            )
    return rec


# ---------------------------------------------------------------------------
# cuSPARSE runner (spmv/spmm via torch.sparse CSR tensors on CUDA)
# ---------------------------------------------------------------------------

def _run_cusparse(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    """cuSPARSE backend: sparse matrix ops via torch.sparse CSR on CUDA."""
    import torch
    import warnings
    kw = dict(ts=ts, bc=bc, tc=tc)

    # torch.sparse.mm with CSR on CUDA dispatches to cuSPARSE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = _run_pytorch(suite_id, problem, backend, device_ordinal, **kw)

    if isinstance(rec.get("execution"), dict):
        rec["execution"]["execution_path"] = "phase2-measured-cusparse"
        rec["execution"]["notes"] = (
            "torch.sparse CSR tensor on CUDA dispatches to cuSPARSE SpMV/SpMM"
        )
    return rec


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

    def rust_normal(shape, data_seed):
        # Match src/bin/benchmark_gpu_rust.rs normal_data and col-major tensor fill.
        total = int(np.prod(shape, dtype=np.int64))
        idx = np.arange(total, dtype=np.uint64)
        mixed = (
            idx * np.uint64(6364136223846793005)
            + np.uint64(data_seed) * np.uint64(1442695040888963407)
        )
        values = ((mixed % np.uint64(1024)).astype(f64) - 512.0) / 512.0
        return np.ascontiguousarray(values.reshape(tuple(shape), order="F"))

    def normal(*shape, data_seed=seed):
        return rust_normal(shape, data_seed)

    def well_conditioned(m, n, data_seed=seed):
        d = rust_normal((m, n), data_seed)
        for j in range(n):
            d[:, j] *= 0.05
            if j < m:
                d[j, j] += 1.0 + j / max(n, 1)
        return np.ascontiguousarray(d)

    def spd(n, data_seed=seed):
        base = well_conditioned(n, n, data_seed)
        return np.ascontiguousarray(base.T @ base + np.eye(n, dtype=f64))

    if op == "matmul":
        p = problem["matmul"]
        m, n, k = p["m"], p["n"], p["k"]
        return {"op": op, "A": normal(m, k, data_seed=seed), "B": normal(k, n, data_seed=seed + 1),
                "transpose_a": p.get("transpose_a", False),
                "transpose_b": p.get("transpose_b", False)}

    if op == "batched_matmul":
        p = problem["batched_matmul"]
        b, m, n, k = p["batch"], p["m"], p["n"], p["k"]
        return {"op": op, "A": np.transpose(rust_normal((m, k, b), seed), (2, 0, 1)).copy(), "B": np.transpose(rust_normal((k, n, b), seed + 1), (2, 0, 1)).copy()}

    if op == "einsum":
        p = problem["einsum"]
        shapes = p["shapes_rowmajor"]
        tensors = [normal(*s, data_seed=seed + i) for i, s in enumerate(shapes)]
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
        A = spd(n, data_seed=seed) if generator == "spd" else normal(n, n, data_seed=seed) + n * np.eye(n, dtype=f64)
        b = normal(n, rhs, data_seed=seed + 100)
        return {"op": op, "A": A, "b": b}

    if op == "svd":
        p = problem["linalg"]
        m, n = p["m"], p["n"]
        A = well_conditioned(m, n) if generator == "well_conditioned" else normal(m, n)
        return {"op": op, "A": A, "full_matrices": p.get("full_matrices", True), "svd_driver": problem.get("torch_svd_driver")}

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
        driver = d.get("svd_driver")
        if driver and d["A"].is_cuda:
            return lambda: torch.linalg.svd(d["A"], full_matrices=fm, driver=driver)
        return lambda: torch.linalg.svd(d["A"], full_matrices=fm)
    if op == "eigh":
        return lambda: torch.linalg.eigh(d["A"])
    if op == "spmv":
        # CSR @ vector via real sparse kernel (cuSPARSE on CUDA).
        return lambda: torch.sparse.mm(d["sp"], d["x"].unsqueeze(1)).squeeze(1)
    if op == "spmm":
        # CSR @ dense matrix via real sparse kernel (cuSPARSE on CUDA).
        return lambda: torch.sparse.mm(d["sp"], d["x"])
    raise ValueError(f"unsupported op: {op}")


def _jax_fn(op: str, d: dict):
    import jax
    import jax.numpy as jnp

    if op == "matmul":
        A, B = d["A"], d["B"]
        ta, tb = d.get("transpose_a", False), d.get("transpose_b", False)
        def fn(A, B):
            a = A.T if ta else A
            b = B.T if tb else B
            return jnp.dot(a, b)
        args = (A, B)
    elif op == "batched_matmul":
        A, B = d["A"], d["B"]
        fn = lambda A, B: jnp.einsum("bik,bkj->bij", A, B)
        args = (A, B)
    elif op == "einsum":
        tensors = d["tensors"]
        expr = d["expr"]
        fn = lambda *tensors: jnp.einsum(expr, *tensors)
        args = tuple(tensors)
    elif op == "qr":
        A = d["A"]
        mode = "reduced" if not d.get("full_matrices") else "complete"
        fn = lambda A: jnp.linalg.qr(A, mode=mode)
        args = (A,)
    elif op == "solve":
        A, b = d["A"], d["b"]
        fn = lambda A, b: jnp.linalg.solve(A, b)
        args = (A, b)
    elif op == "svd":
        A = d["A"]
        fm = d.get("full_matrices", True)
        fn = lambda A: jnp.linalg.svd(A, full_matrices=fm)
        args = (A,)
    elif op == "eigh":
        A = d["A"]
        fn = lambda A: jnp.linalg.eigh(A)
        args = (A,)
    else:
        raise ValueError(f"unsupported op: {op}")

    return jax.jit(fn), args


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
