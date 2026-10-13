#!/usr/bin/env python3
"""CUDA public API fixtures and Python arms for #155; no setup inside timers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

DTYPES = {"bool": "|b1", "i64": "<i8", "f32": "<f4", "f64": "<f8", "c32": "<c8", "c64": "<c16"}
BACKENDS = ["tenferro-cuda-direct", "tenferro-cuda-eager", "pytorch-cuda", "jax-cuda", "jax-cuda-jit"]
RETENTION_BUDGET = 256 << 20


def save_array(root, name, array):
    dtype = next(k for k, v in DTYPES.items() if np.dtype(v) == array.dtype)
    array.ravel(order="F").tofile(root / name)
    return {"file": name, "shape": list(array.shape), "dtype": dtype}


def load_array(root, spec):
    return np.fromfile(root / spec["file"], dtype=DTYPES[spec["dtype"]]).reshape(spec["shape"], order="F")


def make_fixtures(root, suite):
    """Persist exactly the same typed logical values for every backend."""
    root.mkdir(parents=True, exist_ok=False)
    cases = []
    for spec in suite["cases"]:
        if spec["op"] not in {"fft", "ifft", "rfft", "irfft"}:
            from gpu_public_api_ops import fixtures
            inputs, y = fixtures(spec, DTYPES[spec["dtype"]])
            case = dict(spec)
            case.update(inputs=[save_array(root, f'{spec["id"]}_in{i}.bin', x) for i, x in enumerate(inputs)],
                        outputs=[save_array(root, f'{spec["id"]}_out.bin', y)], reference="numpy/scipy double precision",
                        rtol=5e-4 if spec["dtype"].endswith("32") else 1e-9,
                        atol=5e-4 if spec["dtype"].endswith("32") else 1e-9)
            cases.append(case)
            continue
        op, dtype, n = spec["op"], spec["dtype"], spec["n"]
        size = n // 2 + 1 if op == "irfft" else n
        i = np.arange(size, dtype=np.uint64)
        real = ((i * 2654435761 + 7 * 97) % 2001).astype(np.float64) / 2000 - 0.5
        if dtype.startswith("c"):
            imag = ((i * 2654435761 + 8 * 97) % 2001).astype(np.float64) / 2000 - 0.5
            if op == "irfft":
                imag[0] = imag[-1] = 0
            real = real + 1j * imag
        x = real.astype(DTYPES[dtype])
        # Independent double-precision CPU reference; never part of sampling.
        reference_input = x.astype(np.complex128 if np.iscomplexobj(x) else np.float64)
        y = getattr(np.fft, op)(reference_input, n=n, axis=-1, norm="backward")
        output_dtype = ("f" if op == "irfft" else "c") + ("32" if dtype.endswith("32") else "64")
        y = y.astype(DTYPES[output_dtype])
        case = dict(spec)
        # FFT absolute roundoff grows with input energy and transform depth.
        # A fixed elementwise atol rejects cancellation-heavy frequency bins
        # even when their error is negligible relative to transform energy.
        scale = n if op in {"ifft", "irfft"} else 1
        eps = np.finfo(np.float32 if dtype.endswith("32") else np.float64).eps
        fft_atol = max(1e-9, 32 * eps * np.ceil(np.log2(n)) * np.linalg.norm(reference_input) / scale)
        case.update(reference="numpy.fft", inputs=[save_array(root, f'{spec["id"]}_in.bin', x)],
                    outputs=[save_array(root, f'{spec["id"]}_out.bin', y)],
                    rtol=5e-4 if dtype.endswith("32") else 1e-9,
                    atol=float(fft_atol),
                    tolerance_policy="atol=max(1e-9,32*eps*ceil(log2(n))*norm2(input)/normalization); rtol=5e-4(f32),1e-9(f64)")
        cases.append(case)
    (root / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    metadata_path = root.parent / "run.yaml"
    if metadata_path.exists():
        import yaml
        metadata = yaml.safe_load(metadata_path.read_text())
        metadata["timing_policy"].update(
            setup="fixtures, H2D, runtime/session/handles, plans and compilation outside every timer",
            trace_execution="tenferro trace excluded: run_prepared cannot borrow the entered session; JAX JIT compilation outside timing",
            short_operation_sampling="adaptive batches target 2 ms, at most 4096 operations and 256 MiB retained output; output-reuse destinations prepared before timing")
        paths = [Path(".devcontainer/cuda/devcontainer.json"), Path(".devcontainer/cuda/Dockerfile"),
                 Path("src/bin/benchmark_gpu_public_api.rs"), *Path("src/bin/benchmark_gpu_public_api").glob("*.rs"),
                 Path(__file__), Path(__file__).with_name("gpu_public_api_ops.py"),
                 Path("scripts/run_gpu_public_api.sh"), root / "manifest.json",
                 Path(os.environ.get("CARGO_TARGET_DIR", "target")) / "release/benchmark_gpu_public_api"]
        metadata.setdefault("collection", {}).update(
            command=f'BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE={os.environ.get("GPU_BENCH_DEVICE", "0")} BENCH_RUNS={os.environ.get("BENCH_RUNS", "15")} BENCH_WARMUPS={os.environ.get("BENCH_WARMUPS", "3")} ./scripts/run_gpu_public_api.sh {suite["suite_id"].split("/")[1]}',
            devcontainer_config=".devcontainer/cuda/devcontainer.json",
            expected_cases=len(cases), selected_cases=len(cases),
            sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            reference_numpy_version=np.__version__)
        metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))


def verify(actual, expected, case):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"verification shape/dtype mismatch: {actual.shape}/{actual.dtype} vs {expected.shape}/{expected.dtype}")
    equal = actual == expected
    with np.errstate(invalid="ignore"):
        error = np.abs(actual.astype(np.complex128) - expected.astype(np.complex128))
    error = np.where(equal, 0, error)
    scale = case["atol"] + case["rtol"] * np.abs(expected)
    scale = np.where(equal, 1, scale)
    worst = float(np.max(error / scale))
    if not np.isfinite(worst) or worst > 1:
        raise ValueError(f"verification error / tolerance = {worst}")
    return worst


def measure(op, sync, ready, nbytes, warmups, runs, reuse=False):
    def batch(count):
        outputs = [None] * count
        sync()
        start = time.perf_counter_ns()
        for i in range(count):
            outputs[i] = op(i) if reuse else op()
        ready(outputs)
        elapsed = time.perf_counter_ns() - start
        # Keep all outputs alive through synchronization and the stopped clock.
        return elapsed

    first = batch(1)  # explicit lazy initialization, including with zero warmups
    for _ in range(warmups):
        batch(1)
    cap = max(1, min(4096, RETENTION_BUDGET // max(1, nbytes)))
    count = 1
    while batch(count) < 2_000_000 and count < cap:
        count = min(count * 2, cap)
    samples = [batch(count) / 1e6 for _ in range(runs)]
    return {"operations_per_sample": count, "batch_ms": samples,
            "first_run_ms": first / 1e6, "retention_budget_bytes": RETENTION_BUDGET,
            "target_batch_ms": 2, "warmups": warmups,
            "scope": "steady_state_host_api_plus_device_sync"}


def run_python(backend, root, output):
    device = int(os.environ.get("GPU_BENCH_DEVICE", "0"))
    warmups = int(os.environ.get("BENCH_WARMUPS", "3"))
    runs = int(os.environ.get("BENCH_RUNS", "15"))
    if warmups < 0 or runs < 1:
        raise ValueError("invalid warmups/runs")
    if backend == "pytorch-cuda":
        import torch
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        version = torch.__version__
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        gpu = jax.devices("gpu")[device]
        version = jax.__version__
    with output.open("x") as f:
        for case in json.loads((root / "manifest.json").read_text()):
            print(backend, case["id"], flush=True)
            row = {"schema_version": 1, "case_id": case["id"], "op": case["op"],
                   "backend": backend, "version": version, "device_ordinal": device,
                   "setup_policy": "fixtures, H2D, library initialization, compilation and validation outside timing",
                   "session_policy": "backend process and device reused across samples"}
            row["precision_policy"] = "PyTorch TF32 disabled; JAX x64 enabled and matmul precision highest"
            unavailable = {**case.get("excluded", {}), **case.get("unavailable", {})}
            if backend in unavailable:
                row.update(status="unsupported", reason=unavailable[backend])
                f.write(json.dumps(row) + "\n")
                continue
            try:
                hosts = [load_array(root, a) for a in case["inputs"]]
                is_fft = case["op"] in {"fft", "ifft", "rfft", "irfft"}
                x_host = load_array(root, case["inputs"][0])
                expected = load_array(root, case["outputs"][0])
                if backend == "pytorch-cuda":
                    inputs = [torch.from_numpy(np.ascontiguousarray(a) if a.ndim else a).to(f"cuda:{device}") for a in hosts]
                    if is_fft:
                        x = inputs[0]
                        operation = getattr(torch.fft, case["op"])
                        op = lambda: operation(x, n=case["n"], dim=-1, norm="backward")
                    else:
                        from gpu_public_api_ops import torch_operation
                        op = torch_operation(torch, case, inputs)
                    sync = lambda: torch.cuda.synchronize(device)
                    ready = lambda outputs: sync()
                    download = lambda y: y.detach().cpu().numpy()
                    row["synchronization"] = "torch.cuda.synchronize; no D2H in timed region"
                else:
                    inputs = [jax.device_put(a, gpu) for a in hosts]
                    if is_fft:
                        operation = getattr(jnp.fft, case["op"])
                        fn = lambda value: operation(value, n=case["n"], axis=-1, norm="backward")
                    else:
                        from gpu_public_api_ops import jax_operation
                        fn = jax_operation(jax, jnp, case)
                    if backend.endswith("-jit"):
                        fn = jax.jit(fn).lower(*inputs).compile()
                    op = lambda: fn(*inputs)
                    jax.block_until_ready(inputs)
                    sync = lambda: None  # previous batch waits on every output
                    ready = jax.block_until_ready
                    download = np.asarray
                    row["synchronization"] = "jax.block_until_ready on all retained outputs; no D2H in timed region"
                reuse = case["op"].endswith("_into")
                probe = op(0) if reuse else op()
                if backend == "pytorch-cuda":
                    if probe.device.type != "cuda" or probe.device.index != device:
                        raise RuntimeError("CPU fallback or wrong CUDA device")
                elif probe.devices() != {gpu}:
                    raise RuntimeError("CPU fallback or wrong JAX GPU")
                ready([probe])
                error = verify(download(probe), expected, case)
                del probe
                timing = measure(op, sync, ready, expected.nbytes, warmups, runs, reuse=reuse)
                probe = op(timing["operations_per_sample"]-1) if reuse else op()
                ready([probe])
                error = max(error, verify(download(probe), expected, case))
                del probe
                row.update(status="ok", timing=timing,
                           verification={"max_error_over_tolerance": error, "reference": case["reference"]})
            except Exception as error:
                row.update(status="verification_failed" if str(error).startswith("verification") else "runtime_failed", reason=str(error))
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--backend", choices=["prepare", *BACKENDS[2:]], required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    suite = json.loads(args.suite.read_text())
    if args.backend == "prepare":
        make_fixtures(args.fixtures, suite)
    else:
        if args.output is None:
            p.error("--output is required for sampling")
        run_python(args.backend, args.fixtures, args.output)


if __name__ == "__main__":
    main()
