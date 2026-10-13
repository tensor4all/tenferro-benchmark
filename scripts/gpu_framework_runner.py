"""Matched Python CUDA arms for the maintained GPU suites (#155).

Preparation and verification are explicit; timing reuses the bounded batched
sampler used by the public API suites. This module performs no work at import.
"""
from __future__ import annotations

import json
import os
import time

from benchmark_gpu_public_api import measure


def leaves(value):
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in leaves(item)]
    return [value]


def require_verification(result):
    status, absolute, relative = result
    if status != "passed":
        raise ValueError(f"verification {status}: absolute={absolute}, relative={relative}")
    return result


def compiler_options_for(problem, backend):
    # Only explicit JIT compilation accepts these options. Primitive eager JAX
    # dispatch and PyTorch do not use them and must not claim they did.
    if not backend.startswith("jax-cuda") or backend == "jax-cuda-eager":
        return {}
    override = os.environ.get("GPU_BENCH_JAX_COMPILER_OPTIONS")
    options = (json.loads(override) if override is not None else
               dict(problem.get("tensor_network", {}).get("jax_compiler_options", {})))
    if not isinstance(options, dict):
        raise ValueError("JAX compiler options must be a JSON object")
    return options


def run(legacy, suite_id, problem, backend, device_ordinal, *, ts, bc, tc):
    case_start = time.monotonic_ns()
    import numpy as np

    is_jax = backend.startswith("jax-cuda")
    verify_only = os.environ.get("GPU_BENCH_VERIFY_ONLY") == "1"
    op = problem["op"]
    run_spec, verify_spec = problem.get("run", {}), problem.get("verify", {})
    warmups, runs = int(run_spec.get("warmups", 3)), int(run_spec.get("runs", 7))
    rtol, atol = float(verify_spec.get("rtol", 1e-5)), float(verify_spec.get("atol", 1e-8))
    path = "prepared-jax-cuda-jit" if is_jax else "pytorch-cuda-eager"
    if backend == "jax-cuda-tree-jit":
        path = "prepared-jax-cuda-tree-kernels"
    if backend == "jax-cuda-eager":
        path = "prepared-jax-cuda-eager-tree"
    kw = dict(ts=ts, bc=bc, tc=tc)
    compile_ms = None
    compiler_options = compiler_options_for(problem, backend)
    print(backend, problem["id"], "setup", flush=True)
    try:
        if is_jax:
            import jax
            jax.config.update("jax_enable_x64", True)
            jax.config.update("jax_default_matmul_precision", "highest")
            try:
                devices = jax.devices("gpu")
            except RuntimeError as error:
                return legacy._stub(suite_id, problem, backend, device_ordinal, status="not_configured",
                                    reason=str(error), path=path, **kw)
            if device_ordinal < 0 or device_ordinal >= len(devices):
                return legacy._stub(suite_id, problem, backend, device_ordinal, status="not_configured",
                                    reason=f"CUDA ordinal {device_ordinal} is unavailable", path=path, **kw)
            device = devices[device_ordinal]
            version = f"jax={jax.__version__}"
            ready = jax.block_until_ready
            sync = lambda: None  # all input and previous output work is awaited
            download = np.asarray
        else:
            import torch
            torch.set_num_threads(1)
            torch.cuda.set_device(device_ordinal)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            device = torch.device(f"cuda:{device_ordinal}")
            version = f"torch={torch.__version__}"
            sync = lambda: torch.cuda.synchronize(device)
            ready = lambda outputs: sync()
            download = lambda value: value.detach().cpu().numpy()

        backend_end = time.monotonic_ns()

        if op == "tensor_network_contract":
            source, bond_dim, fill = legacy._tensor_network_spec(problem)
            network = legacy.tnc.load_tensor_network(source)
            hosts = [np.full((bond_dim,) * len(ix), fill, dtype=np.float32) for ix in network["inputs"]]
            reference = legacy.tnc.constant_tree_reference(network, bond_dim, fill)
            print(backend, problem["id"], "constant-tree reference complete", flush=True)
            if is_jax:
                inputs = [jax.device_put(value, device) for value in hosts]
                ready(inputs)
                start = time.monotonic_ns()
                if backend == "jax-cuda-eager":
                    prepared = legacy.tnc.prepare_tree(network["tree"], inputs, "jax")
                    operation = lambda: prepared(inputs)
                elif backend == "jax-cuda-tree-jit":
                    prepared = legacy.tnc.prepare_tree_jax_kernels(network["tree"], inputs, compiler_options)
                    operation = lambda: prepared(inputs)
                else:
                    prepared = jax.jit(legacy.tnc.prepare_tree(network["tree"], inputs, "jax"))
                    print(backend, problem["id"], "lowering whole-tree JIT", flush=True)
                    lowered = prepared.lower(inputs)
                    print(backend, problem["id"], "compiling whole-tree JIT", flush=True)
                    compiled = lowered.compile(compiler_options=compiler_options)
                    operation = lambda: compiled(inputs)
                compile_ms = None if backend == "jax-cuda-eager" else (time.monotonic_ns() - start) / 1e6
            else:
                inputs = [torch.from_numpy(value).to(device) for value in hosts]
                prepared = legacy.tnc.prepare_tree(network["tree"], inputs, "torch")
                operation = lambda: prepared(inputs)

            def check(result):
                actual = download(result)
                if actual.dtype != np.float32 or actual.shape != reference.shape or not np.all(np.isfinite(actual)):
                    raise ValueError("verification tensor-network shape/nonfinite output")
                error = np.abs(actual.astype(np.float64) - reference)
                absolute = float(np.max(error))
                relative = float(np.max(error / np.maximum(np.abs(reference), 1e-10)))
                passed = np.all(error <= atol + rtol * np.abs(reference))
                return require_verification(("passed" if passed else "failed", absolute, relative))
        else:
            data = legacy._make_data_np(problem)
            if is_jax:
                inputs = legacy._to_jax(data, device)
                prepared, arguments = legacy._jax_fn(op, inputs)
                ready(arguments)
                start = time.monotonic_ns()
                compiled = prepared.lower(*arguments).compile(compiler_options=compiler_options)
                compile_ms = (time.monotonic_ns() - start) / 1e6
                operation = lambda: compiled(*arguments)
                # Factorizations use reconstruction/residual checks and do not
                # require a redundant CPU factorization of large GPU fixtures.
                reference = None
                if op not in {"qr", "svd", "eigh", "solve"}:
                    cpu_data = legacy._to_jax_cpu(data)
                    cpu_fn, cpu_args = legacy._jax_fn(op, cpu_data)
                    reference = ready(cpu_fn(*cpu_args))
                check = lambda result: require_verification(legacy._verify_jax(op, result, reference, data, rtol, atol))
            else:
                inputs = legacy._to_torch(data, device)
                operation = legacy._torch_fn(op, inputs)
                reference = None
                if op not in {"qr", "svd", "eigh", "solve"}:
                    reference = legacy._torch_fn(op, legacy._to_torch_cpu_f64(data))()
                check = lambda result: require_verification(legacy._verify_torch(op, result, reference, data, rtol, atol))

        def assert_placement(result):
            for value in leaves(result):
                valid = value.devices() == {device} if is_jax else (value.device == device and value.is_cuda)
                if not valid:
                    raise RuntimeError("CPU fallback or wrong CUDA device in output")

        numerical_check = check
        def check(result):
            arrays = [download(value) for value in leaves(result)]
            dtype = np.dtype(np.float32 if op == "tensor_network_contract" else np.float64)
            if any(value.dtype != dtype or not np.all(np.isfinite(value)) for value in arrays):
                raise ValueError("verification output dtype/nonfinite value")
            if op in {"qr", "svd", "eigh"}:
                matrix = data["A"]
                if op == "qr":
                    q, r = arrays
                    if q.ndim != 2 or r.shape != (q.shape[1], matrix.shape[1]) or q.shape[0] != matrix.shape[0]:
                        raise ValueError("verification QR output shapes")
                    orthogonal = [q.T @ q]
                elif op == "svd":
                    u, singular, vh = arrays
                    k = min(matrix.shape)
                    if singular.shape != (k,) or u.shape[0] != matrix.shape[0] or vh.shape[1] != matrix.shape[1]:
                        raise ValueError("verification SVD output shapes")
                    orthogonal = [u.T @ u, vh @ vh.T]
                    if np.any(singular < 0) or np.any(np.diff(singular) > atol):
                        raise ValueError("verification invalid singular values")
                else:
                    eigenvalues, vectors = arrays
                    if eigenvalues.shape != (matrix.shape[0],) or vectors.shape != matrix.shape:
                        raise ValueError("verification eigh output shapes")
                    orthogonal = [vectors.T @ vectors]
                for gram in orthogonal:
                    if not np.allclose(gram, np.eye(gram.shape[0]), rtol=rtol, atol=atol):
                        raise ValueError("verification factor orthogonality")
            elif op != "tensor_network_contract" and reference is not None:
                if len(arrays) != 1 or arrays[0].shape != tuple(reference.shape):
                    raise ValueError("verification output shape")
            elif op == "solve" and arrays[0].shape != data["b"].shape:
                raise ValueError("verification solve output shape")
            return numerical_check(result)

        # Mandatory initialization, placement and numerical check also happen
        # when zero warmups or verification-only execution is requested.
        setup_end = time.monotonic_ns()
        probe = operation()
        ready(probe)
        assert_placement(probe)
        probe_ready = time.monotonic_ns()
        verification = check(probe)
        verified = time.monotonic_ns()
        print(backend, problem["id"], "initial verification passed", flush=True)
        output_bytes = sum(int(value.size * value.dtype.itemsize) if is_jax else value.numel() * value.element_size() for value in leaves(probe))
        del probe
        sampling = None
        final_wall_ns = 0
        if not verify_only:
            sampling = measure(operation, sync, ready, output_bytes, warmups, runs)
            final_start = time.monotonic_ns()
            final = operation()
            ready(final)
            assert_placement(final)
            after = check(final)
            verification = ("passed", max(verification[1], after[1]), max(verification[2], after[2]))
            del final
            final_wall_ns = time.monotonic_ns() - final_start
        stats = (None,) * 5 if verify_only else legacy._timing_stats([value / sampling["operations_per_sample"] for value in sampling["batch_ms"]])
        first, median, minimum, p95, iqr = stats
        record = legacy._ok_record(suite_id, problem, backend, device_ordinal, path=path,
            warmup_runs=warmups, timed_runs=0 if verify_only else runs,
            first_run_ms=first, median_ms=median, min_ms=minimum, p95_ms=p95, iqr_ms=iqr,
            compile_time_ms=compile_ms, ver_status=verification[0], max_abs=verification[1], max_rel=verification[2],
            rtol=rtol, atol=atol, env=legacy._base_env(device_ordinal, ts, bc, tc))
        record["execution"]["notes"] = json.dumps({
            "sampling": sampling, "setup": "inputs/transfers, plans, compilation, lazy initialization and validation outside clocks",
            "precision": "JAX x64/highest; PyTorch TF32 disabled", "framework_version": version,
            "compiler_options": compiler_options if is_jax else None,
            "physical_device_index": os.environ.get("GPU_BENCH_PHYSICAL_DEVICE"),
            "device_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"), "verification_only": verify_only,
            "role": "debug" if verify_only else "benchmark",
            "host_memory_limit_bytes": open("/sys/fs/cgroup/memory.max").read().strip() if os.path.exists("/sys/fs/cgroup/memory.max") else None,
            "wall_phases_ns": {
                "backend_initialization": backend_end-case_start,
                "input_reference_and_preparation": setup_end-backend_end-int((compile_ms or 0)*1e6),
                "compilation": int((compile_ms or 0)*1e6),
                "initialization_execution_and_placement": probe_ready-setup_end,
                "initial_validation": verified-probe_ready,
                "post_sampling_execution_and_validation": final_wall_ns,
                "case_body_before_json_write": time.monotonic_ns()-case_start,
                **({f"sampler_{k}":v for k,v in sampling["wall_phases_ns"].items()} if sampling else {}),
            },
            "wall_phase_scope": "case phases include preparation/cleanup; recorded outside operation clocks; process startup/JSONL writes excluded",
        }, sort_keys=True)
        if op == "tensor_network_contract":
            record["verification"]["reference_backend"] = "analytic_constant_tree_fp64_from_exact_f32_inputs"
        if verify_only:
            record["status"] = "verification_only"  # diagnostic records cannot pass result schema/publication
        return record
    except Exception as error:
        message = str(error)
        status = "verification_failed" if message.startswith("verification") else ("oom" if "out of memory" in message.lower() else "runtime_failed")
        return legacy._stub(suite_id, problem, backend, device_ordinal, status=status, reason=message,
                            path=path, ver_status="failed" if status == "verification_failed" else "skipped", rtol=rtol, atol=atol, **kw)
