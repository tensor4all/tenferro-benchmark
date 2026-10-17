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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import tensornetwork_contract as tnc

# Number of chained elementwise steps in `elementwise_chain` problems; must
# match `ELEMENTWISE_CHAIN_STEPS` in src/bin/benchmark_gpu_rust.rs.
ELEMENTWISE_CHAIN_STEPS = 8

_PYTORCH_OPS = {"matmul", "batched_matmul", "einsum", "elementwise_chain", "tensor_network_contract", "qr", "solve", "svd", "eigh", "spmv", "spmm"}
_JAX_OPS = {"elementwise_chain", "matmul", "batched_matmul", "einsum", "tensor_network_contract", "qr", "solve", "svd", "eigh"}
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
            suite_backends = suite.get("backends", [])
            for prob in suite["problems"]:
                if problem_filter and prob["id"] != problem_filter:
                    continue
                for backend in backends:
                    rec = _run_one(suite["suite_id"], prob, backend, device_ordinal,
                                   suite_backends=suite_backends, ts=ts, bc=bc, tc=tc)
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _backend_candidates(problem, suite_backends):
    candidates = list(problem.get("only_backends") or suite_backends)
    skipped = set(problem.get("skip_backends") or [])
    return [candidate for candidate in candidates if candidate not in skipped]


def _run_one(suite_id, problem, backend, device_ordinal, *, suite_backends=None, ts, bc, tc):
    kw = dict(ts=ts, bc=bc, tc=tc)
    op = problem["op"]
    pid = problem["id"]
    candidates = _backend_candidates(problem, suite_backends or [])
    if backend not in candidates:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="unsupported", reason=f"{backend} not selected for {pid}" + (": " + problem.get("notes", "") if backend in problem.get("skip_backends", []) else ""),
                     path="phase2-runner", **kw)
    if backend == "pytorch-cuda":
        if op not in _PYTORCH_OPS:
            return _stub(suite_id, problem, backend, device_ordinal,
                         status="not_configured", reason=f"pytorch-cuda: op={op} not implemented",
                         path="phase2-runner", **kw)
        if op == "tensor_network_contract":
            return _run_tensor_network_pytorch(suite_id, problem, backend, device_ordinal, **kw)
        return _run_pytorch(suite_id, problem, backend, device_ordinal, **kw)
    if backend in {"jax-cuda", "jax-cuda-tree-jit", "jax-cuda-eager"}:
        if op not in _JAX_OPS:
            return _stub(suite_id, problem, backend, device_ordinal,
                         status="not_configured", reason=f"jax-cuda: op={op} not implemented",
                         path="phase2-runner", **kw)
        if op == "tensor_network_contract":
            return _run_tensor_network_jax(suite_id, problem, backend, device_ordinal, **kw)
        return _run_jax(suite_id, problem, backend, device_ordinal, **kw)
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
    import gpu_framework_runner
    return gpu_framework_runner.run(sys.modules[__name__], suite_id, problem, backend,
                                    device_ordinal, ts=ts, bc=bc, tc=tc)


# ---------------------------------------------------------------------------
# Tensor network runner (TensorNetworkBenchmarks parity)
# ---------------------------------------------------------------------------

def _tensor_network_spec(problem: dict) -> tuple[Path, int, float]:
    tn = problem["tensor_network"]
    source = tnc.resolve_source_path(PROJECT_DIR, tn["source"])
    bond_dim = int(tn.get("bond_dim", tnc.DEFAULT_BOND_DIM))
    fill_value = float(tn.get("fill_value", tnc.build_fill_value()))
    return source, bond_dim, fill_value


def _build_tensor_network_inputs_torch(source: Path, bond_dim: int, fill_value: float, device):
    import torch

    network = tnc.load_tensor_network(source)
    tensors = [
        fill_value
        * torch.ones((bond_dim,) * len(ix), dtype=torch.float32, device=device)
        for ix in network["inputs"]
    ]
    return network, tensors


def _contract_tensor_network_torch(network: dict, tensors, device):
    import torch

    backend = tnc.IntegerLabelEinsumBackend(torch.einsum)
    return tnc.contract_tree(network["tree"], tensors, backend)


def _run_tensor_network_pytorch(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    import gpu_framework_runner
    return gpu_framework_runner.run(sys.modules[__name__], suite_id, problem, backend,
                                    device_ordinal, ts=ts, bc=bc, tc=tc)




def _run_tensor_network_jax(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    import gpu_framework_runner
    return gpu_framework_runner.run(sys.modules[__name__], suite_id, problem, backend,
                                    device_ordinal, ts=ts, bc=bc, tc=tc)


# ---------------------------------------------------------------------------
# JAX runner
# ---------------------------------------------------------------------------

def _run_jax(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    import gpu_framework_runner
    return gpu_framework_runner.run(sys.modules[__name__], suite_id, problem, backend,
                                    device_ordinal, ts=ts, bc=bc, tc=tc)


# ---------------------------------------------------------------------------
# CUTLASS runner (f64 GEMM via JIT-compiled CUTLASS kernel)
# ---------------------------------------------------------------------------

_CUTLASS_EXT = None
_CUTLASS_AVAILABLE = None
_CUTLASS_ERROR = None


def _load_cutlass_ext():
    """JIT-compile and cache the CUTLASS f64 GEMM extension."""
    global _CUTLASS_EXT, _CUTLASS_AVAILABLE, _CUTLASS_ERROR
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
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
using Gemm = cutlass::gemm::device::Gemm<double, cutlass::layout::RowMajor,
    double, cutlass::layout::RowMajor, double, cutlass::layout::RowMajor,
    double, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80>;
using Batched = cutlass::gemm::device::GemmBatched<double, cutlass::layout::RowMajor,
    double, cutlass::layout::RowMajor, double, cutlass::layout::RowMajor,
    double, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80>;
class CutlassPrepared {
    torch::Tensor a_, b_, workspace_;
    Gemm gemm_;
    Gemm::Arguments gemm_arguments_;
    Batched batched_;
    Batched::Arguments batched_arguments_;
    int m_, n_, k_, batch_;
    bool is_batched_;
    std::vector<int64_t> output_shape_;
    void check(cutlass::Status status) {
        TORCH_CHECK(status == cutlass::Status::kSuccess, cutlass::cutlassGetStatusString(status));
    }
    void* workspace() { return workspace_.defined() ? workspace_.data_ptr() : nullptr; }
    Gemm::Arguments args(torch::Tensor const& c) {
        return {{m_, n_, k_}, {a_.data_ptr<double>(), k_}, {b_.data_ptr<double>(), n_},
                {c.data_ptr<double>(), n_}, {c.data_ptr<double>(), n_}, {1.0, 0.0}};
    }
    Batched::Arguments batch_args(torch::Tensor const& c) {
        return {{m_, n_, k_}, {a_.data_ptr<double>(), k_}, a_.stride(0),
                {b_.data_ptr<double>(), n_}, b_.stride(0),
                {c.data_ptr<double>(), n_}, int64_t(m_)*n_,
                {c.data_ptr<double>(), n_}, int64_t(m_)*n_, {1.0, 0.0}, batch_};
    }
public:
    CutlassPrepared(torch::Tensor a, torch::Tensor b): a_(a), b_(b) {
        TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.device() == b.device());
        TORCH_CHECK(a.scalar_type() == torch::kFloat64 && b.scalar_type() == torch::kFloat64);
        TORCH_CHECK(a.is_contiguous() && b.is_contiguous());
        TORCH_CHECK((a.dim() == 2 || a.dim() == 3) && b.dim() == a.dim());
        is_batched_ = a.dim() == 3;
        m_=a.size(-2); k_=a.size(-1); n_=b.size(-1);
        batch_=is_batched_ ? a.size(0) : 1;
        TORCH_CHECK(b.size(-2) == k_ && (!is_batched_ || b.size(0) == batch_));
        output_shape_ = is_batched_ ? std::vector<int64_t>{batch_,m_,n_} : std::vector<int64_t>{m_,n_};
        auto c = torch::empty(output_shape_, a.options());
        auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
        size_t bytes;
        if (is_batched_) { batched_arguments_=batch_args(c); check(batched_.can_implement(batched_arguments_)); bytes=Batched::get_workspace_size(batched_arguments_); }
        else { gemm_arguments_=args(c); check(gemm_.can_implement(gemm_arguments_)); bytes=Gemm::get_workspace_size(gemm_arguments_); }
        if (bytes) workspace_=torch::empty({int64_t(bytes)}, a.options().dtype(torch::kUInt8));
        if (is_batched_) check(batched_.initialize(batched_arguments_,workspace(),stream));
        else check(gemm_.initialize(gemm_arguments_,workspace(),stream));
    }
    torch::Tensor run() {
        auto c = torch::empty(output_shape_, a_.options());
        auto stream = at::cuda::getCurrentCUDAStream(a_.get_device()).stream();
        auto pointer=c.data_ptr<double>();
        if (is_batched_) {
            batched_arguments_.ref_C.reset(pointer); batched_arguments_.ref_D.reset(pointer);
            check(batched_.update(batched_arguments_,workspace())); check(batched_.run(stream));
        } else {
            gemm_arguments_.ref_C.reset(pointer); gemm_arguments_.ref_D.reset(pointer);
            check(gemm_.update(gemm_arguments_,workspace())); check(gemm_.run(stream));
        }
        return c;
    }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<CutlassPrepared>(m,"CutlassPrepared")
        .def(pybind11::init<torch::Tensor,torch::Tensor>()).def("run",&CutlassPrepared::run);
}
"""
    cpp_src = ""
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
        _CUTLASS_ERROR = str(e)
        _CUTLASS_AVAILABLE = False
        _CUTLASS_EXT = None
    return _CUTLASS_EXT


def _sample_vendor(operation, sync, check, device, warmups, runs, phases=None):
    from benchmark_gpu_public_api import measure
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    def tensor(value):
        return value.tensor if hasattr(value, "tensor") else value
    def validate(value):
        value = tensor(value)
        if value.device != device or value.dtype != torch.float64:
            raise RuntimeError("verification: vendor output device/dtype mismatch")
        result = check(value)
        if result[0] != "passed":
            raise RuntimeError(f"verification: {result}")
        return result
    phases = {} if phases is None else phases
    initial_start = time.monotonic_ns()
    probe = operation()
    sync()
    probe_ready = time.monotonic_ns()
    verified = validate(probe)
    phases.update(initialization_execution_and_placement=probe_ready-initial_start,
                  initial_validation=time.monotonic_ns()-probe_ready)
    nbytes = tensor(probe).numel()*tensor(probe).element_size()
    del probe
    if os.environ.get("GPU_BENCH_VERIFY_ONLY") == "1":
        retained = [operation() for _ in range(3)]
        sync()
        checks = [validate(value) for value in retained]
        pointers = [tensor(value).data_ptr() for value in retained if hasattr(tensor(value), "data_ptr")]
        if pointers and len(set(pointers)) != len(retained):
            raise RuntimeError("verification: allocating vendor operation reused a live output")
        return None, ("passed", max([verified[1]]+[value[1] for value in checks]),
                              max([verified[2]]+[value[2] for value in checks]))
    sampling = measure(operation, sync, lambda outputs: sync(), nbytes, warmups, runs)
    final_start = time.monotonic_ns()
    final = operation()
    sync()
    after = validate(final)
    phases.update(post_sampling_execution_and_validation=time.monotonic_ns()-final_start,
                  **{f"sampler_{key}":value for key,value in sampling["wall_phases_ns"].items()})
    return sampling, ("passed",max(verified[1],after[1]),max(verified[2],after[2]))


def _vendor_stats(sampling):
    return ((None,)*5 if sampling is None else
            _timing_stats([value/sampling["operations_per_sample"] for value in sampling["batch_ms"]]))


def _vendor_record(record, sampling, framework, phases=None):
    record["execution"]["notes"] = json.dumps(dict(sampling=sampling,
        setup="inputs, layouts, native operators/workspaces, initialization and validation outside clocks",
        precision="native float64; no dtype conversion in clocks",
        wall_phases_ns=phases or {},
        wall_phase_scope="case phases outside operation clocks; backend initialization includes extension load/compile; interpreter startup and JSON writes excluded",
        framework_version=framework, native_index_dtype="int32" if record["backend"] == "ginkgo" else None, verification_only=sampling is None, device_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"),
        physical_device_index=os.environ.get("GPU_BENCH_PHYSICAL_DEVICE")))
    if sampling is None:
        record["status"] = "verification_only"
    return record


def _run_cutlass(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    """CUTLASS backend: f64 GEMM via JIT-compiled cutlass::gemm::device::Gemm<double,...,Sm80>."""
    case_start = time.monotonic_ns()
    import torch
    kw = dict(ts=ts, bc=bc, tc=tc)

    ext = _load_cutlass_ext()
    if ext is None:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="runtime_failed" if _CUTLASS_ERROR else "not_configured",
                     reason=_CUTLASS_ERROR or "CUTLASS headers not found; set CUTLASS_DIR=/opt/cutlass",
                     path="phase2-runner", **kw)

    op = problem["op"]
    run_spec = problem.get("run", {})
    n_warmup = run_spec.get("warmups", 3)
    n_runs = run_spec.get("runs", 7)
    verify_spec = problem.get("verify", {})
    rtol = float(verify_spec.get("rtol", 1e-5))
    atol = float(verify_spec.get("atol", 1e-8))
    path = "prepared-cutlass-gemm-or-gemm-batched"
    device = torch.device(f"cuda:{device_ordinal}")
    torch.cuda.set_device(device)
    backend_end = time.monotonic_ns()

    try:
        data = _make_data_np(problem)
        gpu_data = _to_torch(data, device)
    except Exception as exc:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="runtime_failed", reason=f"data generation: {exc}", path=path, **kw)

    try:
        if op == "matmul":
            a = gpu_data["A"].T if gpu_data.get("transpose_a", False) else gpu_data["A"]
            b = gpu_data["B"].T if gpu_data.get("transpose_b", False) else gpu_data["B"]
        elif op == "batched_matmul":
            a, b = gpu_data["A"], gpu_data["B"]
        elif op == "einsum" and len(gpu_data["tensors"]) == 2:
            a, b = gpu_data["tensors"]
        else:
            return _stub(suite_id, problem, backend, device_ordinal,
                         status="unsupported", reason="no equivalent CUTLASS operation", path=path, **kw)
        prepared = ext.CutlassPrepared(a.contiguous(), b.contiguous())
        cpu_data = _to_torch_cpu_f64(data)
        reference = _torch_fn(op, cpu_data)()
        check = lambda result: _verify_torch(op, result, reference, data, rtol, atol)
        phases = dict(backend_initialization=backend_end-case_start,
                      input_reference_and_preparation=time.monotonic_ns()-backend_end)
        sampling, verification = _sample_vendor(prepared.run, lambda: torch.cuda.synchronize(device),
                                                 check, device, n_warmup, n_runs, phases=phases)
        ver_status, max_abs, max_rel = verification
    except Exception as exc:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="verification_failed" if str(exc).startswith("verification") else "runtime_failed",
                     reason=str(exc), path=path, **kw)
    first, med, min_t, p95, iqr = _vendor_stats(sampling)

    props = torch.cuda.get_device_properties(device)
    env = _base_env(device_ordinal, ts, bc, tc)
    env.update({
        "gpu_name": props.name,
        "gpu_uuid": str(props.uuid) if hasattr(props, "uuid") else None,
        "gpu_memory_bytes": props.total_memory,
        "cuda_version": torch.version.cuda,
        "framework_version": f"cutlass-v3.7 / torch={torch.__version__}",
    })

    phases["case_body_before_json_write"] = time.monotonic_ns()-case_start
    record = _ok_record(suite_id, problem, backend, device_ordinal,
                      path=path,
                      warmup_runs=n_warmup if sampling else 0, timed_runs=n_runs if sampling else 0,
                      first_run_ms=first, median_ms=med, min_ms=min_t, p95_ms=p95, iqr_ms=iqr,
                      compile_time_ms=None,
                      ver_status=ver_status, max_abs=max_abs, max_rel=max_rel,
                      rtol=rtol, atol=atol,
                      env=env)

    record["execution"]["synchronization"] = "torch.cuda.synchronize"
    return _vendor_record(record, sampling, env["framework_version"], phases)


# ---------------------------------------------------------------------------
# Ginkgo runner (CSR SpMV/SpMM via JIT-compiled extension linking libginkgo)
# ---------------------------------------------------------------------------

_GINKGO_EXT = None
_GINKGO_AVAILABLE = None
_GINKGO_ERROR = None


def _load_ginkgo_ext():
    """JIT-compile and cache the Ginkgo CSR SpMM extension."""
    global _GINKGO_EXT, _GINKGO_AVAILABLE, _GINKGO_ERROR
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
// Retain both the native output descriptor and Tensor through synchronization/clock stop.
struct GinkgoOutput {
    torch::Tensor tensor;
    std::unique_ptr<gko::matrix::Dense<double>> native;
    GinkgoOutput(torch::Tensor value, std::unique_ptr<gko::matrix::Dense<double>> dense)
        : tensor(std::move(value)), native(std::move(dense)) {}
};
class GinkgoSpmm {
    using ValueType = double;
    using IndexType = int;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    using Dense = gko::matrix::Dense<ValueType>;
    torch::Tensor row_ptrs_, col_idxs_, values_, x_;
    std::shared_ptr<gko::CudaExecutor> exec_;
    std::unique_ptr<Csr> matrix_;
    std::unique_ptr<Dense> input_;
    int64_t rows_, rhs_;
    gko::dim<2> output_size_;
    std::vector<int64_t> output_shape_;
public:
    GinkgoSpmm(torch::Tensor row_ptrs, torch::Tensor col_idxs,
               torch::Tensor values, torch::Tensor x, int64_t rows, int64_t cols)
        : row_ptrs_(row_ptrs), col_idxs_(col_idxs), values_(values), x_(x),
          rows_(rows), rhs_(x.numel() / cols), output_size_(rows_,rhs_) {
        output_shape_=x_.dim()==1 ? std::vector<int64_t>{rows_} : std::vector<int64_t>{rows_,rhs_};
        TORCH_CHECK(row_ptrs.is_cuda() && col_idxs.is_cuda() && values.is_cuda() && x.is_cuda(),
                    "GinkgoSpmm requires CUDA inputs");
        exec_ = gko::CudaExecutor::create(x.get_device(), gko::ReferenceExecutor::create());
        matrix_ = Csr::create(exec_, gko::dim<2>{(gko::size_type)rows, (gko::size_type)cols},
            gko::make_array_view(exec_, values.numel(), values.data_ptr<ValueType>()),
            gko::make_array_view(exec_, col_idxs.numel(), col_idxs.data_ptr<IndexType>()),
            gko::make_array_view(exec_, rows + 1, row_ptrs.data_ptr<IndexType>()));
        input_ = Dense::create(exec_, gko::dim<2>{(gko::size_type)cols, (gko::size_type)rhs_},
            gko::make_array_view(exec_, x.numel(), x.data_ptr<ValueType>()), rhs_);
        exec_->synchronize();
    }
    void synchronize() { exec_->synchronize(); }
    std::shared_ptr<GinkgoOutput> run() {
        // Only the operation's fresh output is allocated in the measured call.
        auto y = torch::empty(output_shape_, x_.options());
        auto output = Dense::create(exec_, output_size_,
            gko::make_array_view(exec_, y.numel(), y.data_ptr<ValueType>()), rhs_);
        matrix_->apply(input_.get(), output.get());
        return std::make_shared<GinkgoOutput>(std::move(y), std::move(output));
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<GinkgoOutput, std::shared_ptr<GinkgoOutput>>(m, "GinkgoOutput")
        .def_readonly("tensor", &GinkgoOutput::tensor);
    pybind11::class_<GinkgoSpmm>(m, "GinkgoSpmm")
        .def(pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t>())
        .def("run", &GinkgoSpmm::run).def("synchronize", &GinkgoSpmm::synchronize);
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
    except Exception as e:
        _GINKGO_ERROR = str(e)
        _GINKGO_AVAILABLE = False
        _GINKGO_EXT = None
    return _GINKGO_EXT


def _run_ginkgo(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    """Ginkgo backend: CSR SpMV/SpMM via libginkgo CUDA executor."""
    case_start = time.monotonic_ns()
    import torch
    kw = dict(ts=ts, bc=bc, tc=tc)

    ext = _load_ginkgo_ext()
    if ext is None:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="runtime_failed" if _GINKGO_ERROR else "not_configured",
                     reason=_GINKGO_ERROR or "Ginkgo library not found; set GINKGO_DIR=/opt/ginkgo",
                     path="phase2-runner", **kw)

    op = problem["op"]
    run_spec = problem.get("run", {})
    n_warmup = run_spec.get("warmups", 3)
    n_runs = run_spec.get("runs", 7)
    verify_spec = problem.get("verify", {})
    rtol = float(verify_spec.get("rtol", 1e-5))
    atol = float(verify_spec.get("atol", 1e-8))
    path = "prepared-ginkgo-csr-apply"
    device = torch.device(f"cuda:{device_ordinal}")
    torch.cuda.set_device(device)
    backend_end = time.monotonic_ns()

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

        x_np = data["x"]
        x = torch.as_tensor(x_np, dtype=torch.float64, device=device).contiguous()
        prepared = ext.GinkgoSpmm(crow, ccol, cval, x, rows, cols)
        reference = torch.sparse.mm(csr, torch.as_tensor(x_np, dtype=torch.float64).reshape(cols, -1)).numpy().ravel()
        expected_shape = (rows,) if op == "spmv" else (rows, x.shape[1])
        def check(result):
            import numpy as np
            if tuple(result.shape) != expected_shape:
                raise RuntimeError("verification: wrong Ginkgo output shape")
            actual = result.detach().cpu().numpy().ravel()
            max_abs = float(np.max(np.abs(actual-reference)))
            max_rel = float(np.max(np.abs(actual-reference)/np.maximum(np.abs(reference),1e-10)))
            passed = np.isfinite(actual).all() and max_abs <= atol+rtol*max(float(np.max(np.abs(reference))),1.0)
            return ("passed" if passed else "failed", max_abs, max_rel)
        phases = dict(backend_initialization=backend_end-case_start,
                      input_reference_and_preparation=time.monotonic_ns()-backend_end)
        sampling, verification = _sample_vendor(prepared.run, prepared.synchronize, check, device, n_warmup, n_runs, phases=phases)
        ver_status, max_abs, max_rel = verification
    except Exception as exc:
        return _stub(suite_id, problem, backend, device_ordinal,
                     status="verification_failed" if str(exc).startswith("verification") else "runtime_failed",
                     reason=str(exc), path=path, **kw)
    first, med, min_t, p95, iqr = _vendor_stats(sampling)
    props = torch.cuda.get_device_properties(device)
    env = _base_env(device_ordinal, ts, bc, tc)
    env.update({
        "gpu_name": props.name,
        "gpu_memory_bytes": props.total_memory,
        "cuda_version": torch.version.cuda,
        "framework_version": "ginkgo-v1.8",
    })

    phases["case_body_before_json_write"] = time.monotonic_ns()-case_start
    record = _ok_record(suite_id, problem, backend, device_ordinal,
                      path=path,
                      warmup_runs=n_warmup if sampling else 0, timed_runs=n_runs if sampling else 0,
                      first_run_ms=first, median_ms=med, min_ms=min_t, p95_ms=p95, iqr_ms=iqr,
                      compile_time_ms=None,
                      ver_status=ver_status, max_abs=max_abs, max_rel=max_rel,
                      rtol=rtol, atol=atol,
                      env=env)

    record["execution"]["synchronization"] = "Ginkgo Executor::synchronize"
    return _vendor_record(record, sampling, env["framework_version"], phases)


# ---------------------------------------------------------------------------
# cuBLAS LT runner (matmul/batched_matmul/einsum via torch.mm on CUDA)
# ---------------------------------------------------------------------------

def _annotate_vendor(record, path, description):
    execution = record.get("execution")
    if isinstance(execution, dict):
        notes = json.loads(execution.get("notes") or "{}")
        if not isinstance(notes, dict):
            raise ValueError("vendor annotation requires structured sampling notes")
        notes["vendor_path"] = description
        execution["execution_path"] = path
        execution["notes"] = json.dumps(notes)
    return record


def _run_cublaslt(suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    """PyTorch GEMM with cuBLASLt preferred; the framework may fall back."""
    import torch
    kw = dict(ts=ts, bc=bc, tc=tc)
    previous = torch.backends.cuda.preferred_blas_library()
    try:
        torch.backends.cuda.preferred_blas_library("cublaslt")
        rec = _run_pytorch(suite_id, problem, backend, device_ordinal, **kw)
    finally:
        torch.backends.cuda.preferred_blas_library(previous)
    return _annotate_vendor(rec, "pytorch-cublaslt-preferred",
                            "PyTorch preferred_blas_library=cublaslt; fallback remains possible; not a raw cuBLASLt API")


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

    return _annotate_vendor(rec, "pytorch-cusolver-preferred",
        "torch.linalg preferred_linalg_library=cusolver; SVD driver=gesvd; not a raw cuSOLVER API")


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

    return _annotate_vendor(rec, "pytorch-csr-cusparse",
        "PyTorch native CSR mv/mm CUDA dispatch; not a raw cuSPARSE API")


# ---------------------------------------------------------------------------
# Data generation (numpy-based, framework-agnostic)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Synthetic sparse-matrix cache
#
# The large synthetic CSR fixtures are deterministic, but each backend runs in
# its own process (scripts/run_gpu_suite.sh), so without a cache every backend
# regenerates the same multi-GB CSR matrix. Cache it keyed by problem id + seed
# + shape under GPU_BENCH_SPARSE_CACHE_DIR (set by the runner to
# <run-dir>/cache) or data/cache/gpu_sparse, and verify the stored identity
# before reuse. Generation and cache I/O happen outside any timed region.
# ---------------------------------------------------------------------------

def _sparse_cache_dir() -> Path:
    override = os.environ.get("GPU_BENCH_SPARSE_CACHE_DIR")
    if override:
        return Path(override)
    return PROJECT_DIR / "data" / "cache" / "gpu_sparse"


def _sparse_cache_meta(problem: dict) -> dict[str, int]:
    spec = problem["sparse"]
    return {
        "seed": int(problem.get("data", {}).get("seed", 0)),
        "rows": int(spec["rows"]),
        "cols": int(spec["cols"]),
        "nnz": int(spec["nnz"]),
        "rhs_cols": int(spec.get("rhs_cols", 1)) if problem["op"] == "spmm" else 0,
    }


def _sparse_cache_path(problem: dict) -> Path:
    meta = _sparse_cache_meta(problem)
    name = (f"{problem['id']}_seed{meta['seed']}_{meta['rows']}x{meta['cols']}"
            f"_nnz{meta['nnz']}.npz")
    return _sparse_cache_dir() / name


def _load_sparse_cache(problem: dict):
    import numpy as np
    path = _sparse_cache_path(problem)
    if not path.exists():
        return None
    meta = _sparse_cache_meta(problem)
    try:
        with np.load(path) as archive:
            stored = {key: int(archive[key].item())
                      for key in ("seed", "rows", "cols", "nnz", "rhs_cols")}
            if stored != meta:
                return None
            row_idx = archive["row_idx"]
            col_idx = archive["col_idx"]
            values = archive["values"]
            x = archive["x"]
    except (OSError, KeyError, ValueError):
        return None
    if row_idx.shape != (meta["nnz"],) or col_idx.shape != (meta["nnz"],) or values.shape != (meta["nnz"],):
        return None
    return {"op": problem["op"], "rows": meta["rows"], "cols": meta["cols"],
            "row_idx": row_idx, "col_idx": col_idx, "values": values, "x": x}


def _store_sparse_cache(problem: dict, data: dict) -> None:
    import numpy as np
    path = _sparse_cache_path(problem)
    meta = _sparse_cache_meta(problem)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        row_idx=data["row_idx"], col_idx=data["col_idx"], values=data["values"], x=data["x"],
        seed=np.array(meta["seed"]), rows=np.array(meta["rows"]), cols=np.array(meta["cols"]),
        nnz=np.array(meta["nnz"]), rhs_cols=np.array(meta["rhs_cols"]),
    )
    os.replace(tmp, path)


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
            + np.uint64((int(data_seed) * 1442695040888963407) & ((1 << 64) - 1))
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

    if op == "elementwise_chain":
        p = problem["elementwise_chain"]
        n = p["n"]
        return {"op": op, "A": normal(n, data_seed=seed), "B": normal(n, data_seed=seed + 1)}

    if op == "qr":
        p = problem["linalg"]
        m, n = p["m"], p["n"]
        A = well_conditioned(m, n) if generator == "well_conditioned" else normal(m, n)
        return {"op": op, "A": A, "full_matrices": p.get("full_matrices", True)}

    if op == "solve":
        p = problem["linalg"]
        n = p["n"]
        rhs = p.get("rhs_cols", 1)
        if generator == "spd":
            A = spd(n, data_seed=seed)
        elif generator == "well_conditioned":
            A = well_conditioned(n, n, data_seed=seed)
        else:
            A = normal(n, n, data_seed=seed) + n * np.eye(n, dtype=f64)
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
        cached = _load_sparse_cache(problem)
        if cached is not None:
            return cached
        p = problem["sparse"]
        rows, cols, nnz = p["rows"], p["cols"], p["nnz"]
        rhs_cols = p.get("rhs_cols", 1) if op == "spmm" else None
        # Generate random CSR sparse matrix with given dimensions
        row_idx = rng.integers(0, rows, nnz)
        col_idx = rng.integers(0, cols, nnz)
        values = rng.standard_normal(nnz).astype(f64)
        x = rng.standard_normal(cols).astype(f64) if op == "spmv" else rng.standard_normal((cols, rhs_cols)).astype(f64)
        data = {"op": op, "rows": rows, "cols": cols,
                "row_idx": row_idx, "col_idx": col_idx, "values": values, "x": x}
        _store_sparse_cache(problem, data)
        return data

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
    if op == "elementwise_chain":
        a, b = d["A"], d["B"]

        def chain():
            t = a
            for _ in range(ELEMENTWISE_CHAIN_STEPS):
                t = torch.tanh(t * a + b)
            return t

        return chain
    if op == "einsum":
        import opt_einsum as oe
        tensors = d["tensors"]
        expression = oe.contract_expression(d["expr"], *(t.shape for t in tensors), optimize="auto")
        return lambda: expression(*tensors, backend="torch")
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
        return lambda: torch.mv(d["sp"], d["x"])
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
    elif op == "elementwise_chain":
        def fn(a, b):
            value = a
            for _ in range(ELEMENTWISE_CHAIN_STEPS):
                value = jnp.tanh(value * a + b)
            return value
        args = (d["A"], d["B"])
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
        if op == "solve":
            x = to_np(gpu_result)
            A = to_np(data["A"])
            b = to_np(data["b"])
            residual = A @ x - b
            b_norm = np.max(np.abs(b))
            a = float(np.max(np.abs(residual)))
            r = float(np.max(np.abs(residual) / max(float(b_norm), 1e-10)))
            return check(a, r, b_norm)

        if op in ("matmul", "batched_matmul", "einsum", "elementwise_chain"):
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
        if op == "solve":
            x = to_np(gpu_result)
            A = data["A"].astype(np.float64)
            b = data["b"].astype(np.float64)
            residual = A @ x - b
            b_norm = np.max(np.abs(b))
            a = float(np.max(np.abs(residual)))
            r = float(np.max(np.abs(residual) / max(float(b_norm), 1e-10)))
            return check(a, r, b_norm)

        if op in ("matmul", "batched_matmul", "einsum", "elementwise_chain"):
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
