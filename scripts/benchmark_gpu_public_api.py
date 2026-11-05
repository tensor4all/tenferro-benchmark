#!/usr/bin/env python3
"""CUDA public API fixtures and Python arms for #155; no setup inside timers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

# The sibling modules are imported by name because this file also runs as a script
# from `scripts/`; the path insert keeps an importing test from failing on them.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gpu_case_traffic import case_traffic  # noqa: E402

DTYPES = {"bool": "|b1", "i32": "<i4", "i64": "<i8", "f32": "<f4", "f64": "<f8", "c32": "<c8", "c64": "<c16"}
BACKENDS = ["tenferro-cuda-direct", "tenferro-cuda-eager", "pytorch-cuda", "jax-cuda", "jax-cuda-jit"]
RETENTION_BUDGET = 256 << 20


def save_array(root, name, array):
    dtype = next(k for k, v in DTYPES.items() if np.dtype(v) == array.dtype)
    array.ravel(order="F").tofile(root / name)
    return {"file": name, "shape": list(array.shape), "dtype": dtype}


def load_array(root, spec):
    return np.fromfile(root / spec["file"], dtype=DTYPES[spec["dtype"]]).reshape(spec["shape"], order="F")


FINGERPRINT_SCHEMA_VERSION = 1
# The seeded weight vector's splitmix64 constants, shared with `src/fingerprint.rs`;
# `scripts/ci/tests/test_fingerprint_parity.py` pins both to one reference array.
FINGERPRINT_SEED = 0x9E3779B97F4A7C15
_U64 = (1 << 64) - 1


def seeded_weights(count):
    """The fixed weight vector in [-1, 1), one weight per element in file order.

    Real weights, not `±1` signs: two elements that share a weight cannot be told
    apart by the weighted dot product, and a sign vector gives half of all pairs the
    same weight. With `±1` signs the first four weights were [-1, +1, +1, -1], so
    [1, 2, 3, 4] and [1, 3, 2, 4] had the same dot product.
    """
    state = 0
    out = np.empty(count, dtype=np.float64)
    for index in range(count):
        state = (state + FINGERPRINT_SEED) & _U64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
        z ^= z >> 31
        # 53 bits scaled into [-1, 1): exact in f64, matching the Rust step for step.
        out[index] = (z >> 11) / (1 << 52) - 1.0
    return out


def fingerprint(array):
    """Five invariants of one expected output, in the harness's document shape."""
    flat = np.asarray(array).ravel(order="F")
    if np.iscomplexobj(flat):
        real = flat.real.astype(np.float64)
        imaginary = flat.imag.astype(np.float64)
    else:
        real = flat.astype(np.float64)
        imaginary = None
    magnitude = np.hypot(real, imaginary) if imaginary is not None else np.abs(real)
    weights = seeded_weights(real.size)
    return {
        "sum_re": float(real.sum()),
        "sum_im": float(imaginary.sum()) if imaginary is not None else 0.0,
        "sum_abs": float(magnitude.sum()),
        "sum_squares": float(np.dot(real, real) + (np.dot(imaginary, imaginary) if imaginary is not None else 0.0)),
        "max_abs": float(magnitude.max()) if magnitude.size else 0.0,
        "dot_seeded_re": float(np.dot(weights, real)),
        "dot_seeded_im": float(np.dot(weights, imaginary)) if imaginary is not None else 0.0,
        "elements": int(real.size),
    }


FINGERPRINT_UNAVAILABLE = ("the expected values are not all finite, so the invariants are not "
                           "representable in JSON")


def save_fingerprint(root, spec, index, case_id):
    """Store one output's invariants next to its fixture.

    `masked_log_softmax` writes `-inf` for a masked element, and its invariants are then
    infinite or NaN. JSON cannot carry those, so the document records that it has no
    invariants and why; the arm then runs no fingerprint diagnostic for that output
    instead of failing to parse the document.
    """
    block = fingerprint(load_array(root, spec))
    available = all(math.isfinite(value) for value in block.values())
    document = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "case_id": case_id,
        "output": index,
        "dtype": spec["dtype"],
        "shape": list(spec["shape"]),
        "fingerprint": block if available else None,
        "unavailable_reason": None if available else FINGERPRINT_UNAVAILABLE,
    }
    (root / f'{spec["file"]}.fingerprint.json').write_text(json.dumps(document, indent=1) + "\n")


def make_fixtures(root, suite):
    """Persist exactly the same typed logical values for every backend."""
    root.mkdir(parents=True, exist_ok=False)
    cases = []
    for spec in suite["cases"]:
        if spec["op"] not in {"fft", "ifft", "rfft", "irfft"}:
            from gpu_public_api_ops import fixtures
            from gpu_multi_output_verification import leaves
            inputs, y = fixtures(spec, DTYPES[spec["dtype"]])
            case = dict(spec)
            case.update(inputs=[save_array(root, f'{spec["id"]}_in{i}.bin', x) for i, x in enumerate(inputs)],
                        outputs=[save_array(root, f'{spec["id"]}_out{i}.bin', v) for i, v in enumerate(leaves(y))], reference="numpy/scipy double precision; all requested outputs",
                        rtol=5e-4 if spec["dtype"].endswith("32") else 1e-9,
                        atol=5e-4 if spec["dtype"].endswith("32") else 1e-9,
                        traffic=case_traffic(inputs, leaves(y), op=spec["op"]))
            if spec.get("full_reduction") and spec["op"] in {"reduce_sum", "reduce_prod"}:
                # Parallel accumulation has an operation-size-dependent
                # roundoff budget. Use exact fixture values, never measured
                # backend discrepancies, to determine this policy.
                eps = np.finfo(inputs[0].dtype).eps
                if spec["op"] == "reduce_sum":
                    case["atol"] = max(case["atol"], float(
                        32 * eps * max(1, np.ceil(np.log2(spec["n"])))
                        * np.linalg.norm(inputs[0].astype(np.float64))))
                    case["tolerance_policy"] = "sum atol=max(base,32*eps*ceil(log2(n))*norm2(input)); base rtol"
                else:
                    case["rtol"] = max(case["rtol"], float(32 * eps * np.sqrt(spec["n"])))
                    case["tolerance_policy"] = "prod rtol=max(base,32*eps*sqrt(n)); base atol"
            if spec.get("full_reduction") and spec["op"] == "reduce_mean":
                # The mean of a cancellation-heavy vector can be tiny. A
                # generic elementwise atol would even accept an all-zero result.
                eps = np.finfo(inputs[0].dtype).eps
                case["atol"] = float(32 * eps * max(1, np.ceil(np.log2(spec["n"])))
                                     * np.linalg.norm(inputs[0].astype(np.float64)) / spec["n"])
                case["tolerance_policy"] = "mean atol=32*eps*ceil(log2(n))*norm2(input)/n; base rtol"
            if spec["dtype"] in {"i32", "i64"}:
                case.update(rtol=0.0, atol=0.0, tolerance_policy="exact integer equality; no floating conversion",
                            reference="NumPy exact integer arithmetic; truncating div/rem; all requested outputs")
            for index, saved in enumerate(case["outputs"]):
                save_fingerprint(root, saved, index, spec["id"])
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
                    traffic=case_traffic([x], [y], op=op),
                    rtol=5e-4 if dtype.endswith("32") else 1e-9,
                    atol=float(fft_atol),
                    tolerance_policy="atol=max(1e-9,32*eps*ceil(log2(n))*norm2(input)/normalization); rtol=5e-4(f32),1e-9(f64)")
        for index, saved in enumerate(case["outputs"]):
            save_fingerprint(root, saved, index, spec["id"])
        cases.append(case)
    (root / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    metadata_path = root.parent / "run.yaml"
    if metadata_path.exists():
        import yaml
        metadata = yaml.safe_load(metadata_path.read_text())
        metadata["timing_policy"].update(
            setup="fixtures, H2D, runtime/session/handles, plans and compilation outside every timer",
            trace_execution="tenferro trace excluded: this suite compares immediate per-op calls, while trace executes a prepared graph in one caller-held session (run_prepared_in); JAX JIT compilation outside timing",
            short_operation_sampling="adaptive batches target 2 ms, at most 4096 operations and 256 MiB retained output; output-reuse destinations prepared before timing")
        verify_only = os.environ.get("GPU_API_VERIFY_ONLY") == "1"
        physical_device = os.environ.get("GPU_BENCH_PHYSICAL_DEVICE", os.environ.get("GPU_BENCH_DEVICE", "0"))
        metadata["gpu_assignment"] = {
            "role": os.environ.get("GPU_API_ROLE", "benchmark"),
            "physical_device_index": int(physical_device),
            "device_uuid": os.environ.get("GPU_BENCH_DEVICE_UUID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "process_device_ordinal": int(os.environ.get("GPU_BENCH_DEVICE", "0")),
            "verification_only": verify_only,
        }
        if verify_only:
            metadata["timing_policy"] = {"scope": "numerical verification only; no calibration or timing samples"}
        paths = [Path(".devcontainer/cuda/devcontainer.json"), Path(".devcontainer/cuda/Dockerfile"),
                 Path("src/bin/benchmark_gpu_public_api.rs"), *Path("src/bin/benchmark_gpu_public_api").glob("*.rs"),
                 Path(__file__), Path(__file__).with_name("gpu_public_api_ops.py"),
                 Path("scripts/run_gpu_public_api.sh"), root / "manifest.json",
                 Path(os.environ.get("CARGO_TARGET_DIR", "target")) / "release/benchmark_gpu_public_api"]
        metadata.setdefault("collection", {}).update(
            command=f'BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE={physical_device} BENCH_RUNS={os.environ.get("BENCH_RUNS", "15")} BENCH_WARMUPS={os.environ.get("BENCH_WARMUPS", "3")} ./scripts/run_gpu_public_api.sh {suite["suite_id"].split("/")[1]}' + (' --verify-only' if verify_only else ''),
            devcontainer_config=".devcontainer/cuda/devcontainer.json",
            expected_cases=len(cases), selected_cases=len(cases),
            sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            reference_numpy_version=np.__version__)
        metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))


def verify(actual, expected, case):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"verification shape/dtype mismatch: {actual.shape}/{actual.dtype} vs {expected.shape}/{expected.dtype}")
    if np.issubdtype(expected.dtype, np.integer):
        if not np.array_equal(actual, expected):
            raise ValueError("verification exact integer mismatch")
        return 0.0
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


def load_fingerprint(root, spec):
    path = root / f'{spec["file"]}.fingerprint.json'
    if not path.exists():
        return None
    document = json.loads(path.read_text())
    if document.get("schema_version") != FINGERPRINT_SCHEMA_VERSION:
        return None
    return document["fingerprint"]


def fingerprint_matches(actual, stored, case):
    """Whether `actual` agrees with a stored fingerprint, in the Rust module's bounds.

    A diagnostic agreement test, never an acceptance test: `verify_outputs` below
    compares every element whatever this returns, because one large element can pay
    the whole allowance a sum is compared with.
    """
    live = fingerprint(actual)
    if live["elements"] != stored["elements"]:
        return False
    count = float(live["elements"])
    absolute = case["atol"] * count
    mass = absolute + case["rtol"] * stored["sum_abs"]
    peaks = 2.0 * stored["max_abs"] * mass

    def close(live_value, stored_value, scale):
        return abs(live_value - stored_value) <= scale

    return (close(live["sum_re"], stored["sum_re"], mass)
            and close(live["sum_im"], stored["sum_im"], mass)
            and close(live["sum_abs"], stored["sum_abs"], mass)
            and close(live["sum_squares"], stored["sum_squares"],
                       absolute + case["rtol"] * stored["sum_squares"] + peaks)
            and close(live["max_abs"], stored["max_abs"], case["atol"] + case["rtol"] * stored["max_abs"])
            and close(live["dot_seeded_re"], stored["dot_seeded_re"], mass)
            and close(live["dot_seeded_im"], stored["dot_seeded_im"], mass))


def fingerprint_diagnostic(values, stored_fingerprints, case):
    """How the recorded fingerprint compares, as (compares, disagreements)."""
    compares = disagreements = 0
    for actual, stored in zip(values, stored_fingerprints):
        if stored is None:
            continue
        compares += 1
        if not fingerprint_matches(actual, stored, case):
            disagreements += 1
    return compares, disagreements


def measure(op, sync, ready, nbytes, warmups, runs, reuse=False, *, synchronize_each_operation=False):
    if getattr(op, "verification_only", False):
        raise ValueError("Ref completion synchronization is not validated for timing")
    def batch(count):
        outputs = [None] * count
        completion_tokens = [None] * count if synchronize_each_operation else None
        if hasattr(op, "prepare"):
            op.prepare(count)
        sync()
        start = time.perf_counter_ns()
        for i in range(count):
            outputs[i] = op(i) if reuse else op()
            if synchronize_each_operation:
                completion_tokens[i] = ready(outputs[i])
        if not synchronize_each_operation:
            completion_tokens = ready(outputs)
        elapsed = time.perf_counter_ns() - start
        # Keep all outputs alive through synchronization and the stopped clock.
        return elapsed

    phase_start = time.monotonic_ns()
    first = batch(1)  # explicit lazy initialization, including with zero warmups
    probe_end = time.monotonic_ns()
    for _ in range(warmups):
        batch(1)
    warmup_end = time.monotonic_ns()
    bytes_per_operation = max(1, nbytes + getattr(op, "synchronization_bytes_per_operation", 0))
    cap = max(1, min(4096, RETENTION_BUDGET // bytes_per_operation))
    count = 1
    while batch(count) < 2_000_000 and count < cap:
        count = min(count * 2, cap)
    calibration_end = time.monotonic_ns()
    samples = [batch(count) / 1e6 for _ in range(runs)]
    sampling_end = time.monotonic_ns()
    return {"operations_per_sample": count, "batch_ms": samples,
            "first_run_ms": first / 1e6, "retention_budget_bytes": RETENTION_BUDGET,
            "retained_bytes_per_operation": bytes_per_operation,
            "effective_retention_bound_bytes": max(RETENTION_BUDGET, bytes_per_operation),
            "single_output_exceeds_configured_budget": bytes_per_operation > RETENTION_BUDGET,
            "target_batch_ms": 2, "warmups": warmups,
            "scope": "steady_state_host_api_plus_device_sync",
            "synchronization_policy": "after_each_operation" if synchronize_each_operation else "after_batch",
            "completion_token_bytes_per_operation": getattr(op, "synchronization_bytes_per_operation", 0),
            "wall_phases_ns": {"initial_probe": probe_end-phase_start,
                               "warmup": warmup_end-probe_end,
                               "calibration": calibration_end-warmup_end,
                               "sampling": sampling_end-calibration_end},
            "wall_phase_scope": "end-to-end phase wall time, including untimed batch preparation/cleanup; outside operation clocks"}


def verify_device(outputs, backend, device):
    """Check every output leaf before any verification download."""
    from gpu_multi_output_verification import leaves
    for value in leaves(outputs):
        if backend == "pytorch-cuda":
            if value.device.type != "cuda" or value.device.index != device:
                raise RuntimeError("CPU fallback or wrong CUDA device")
        elif value.devices() != {device}:
            raise RuntimeError("CPU fallback or wrong JAX GPU")


OVERHEAD_BUDGET = float(os.environ.get("GPU_API_OVERHEAD_BUDGET", "0.2"))


def overhead_warning(case_id, timed_ms, case_wall_ms, budget):
    """The overhead line a case records when it spent more than `budget` outside its
    timed region, or None when it did not.

    The budget has no measured basis: it labels the untimed split for the report. It
    is recorded in the row and printed, and nothing reads it back into a status or an
    exit code, because a case that spends a long time setting up is still a
    measurement.
    """
    if not timed_ms or timed_ms <= 0:
        return None
    untimed = max(0.0, case_wall_ms - timed_ms)
    if untimed <= budget * timed_ms:
        return None
    return (f'{case_id} untimed {untimed:.1f} ms is '
            f'{100 * untimed / timed_ms:.0f}% of the {timed_ms:.1f} ms '
            f'timed region (budget {100 * budget:.0f}%)')


def run_python(backend, root, output):
    device = int(os.environ.get("GPU_BENCH_DEVICE", "0"))
    verify_only = os.environ.get("GPU_API_VERIFY_ONLY") == "1"
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
    overhead_budget = OVERHEAD_BUDGET
    with output.open("x") as f:
        for case in json.loads((root / "manifest.json").read_text()):
            print(backend, case["id"], flush=True)
            row = {"schema_version": 1, "case_id": case["id"], "op": case["op"],
                   "backend": backend, "version": version, "device_ordinal": device,
                   "gpu_role": os.environ.get("GPU_API_ROLE", "benchmark"),
                   "physical_device_index": int(os.environ.get("GPU_BENCH_PHYSICAL_DEVICE", str(device))),
                   "device_uuid": os.environ.get("GPU_BENCH_DEVICE_UUID"),
                   "traffic": case.get("traffic") or {
                       "bytes_per_operation": None,
                       "unknown_reason": "fixture manifest carries no traffic model for this case"},
                   "setup_policy": "fixtures, H2D, library initialization, compilation and validation outside timing",
                   "session_policy": "backend process and device reused across samples"}
            row["precision_policy"] = "PyTorch TF32 disabled; JAX x64 enabled and matmul precision highest"
            if case["op"].startswith("einsum_plan"):
                row["reference_api"] = "native matrix multiplication/accumulation with matched outputs, not a prepared-plan interface"
                row["input_layout_policy"] = "native input representation; transpose views/values prepared outside timing when selected"
            if case["op"] in {"dot_general_axes", "dot_general_conj"}:
                row["dot_configuration"] = {key: case.get(key, False if key.endswith("conj") else []) for key in ("lhs_contracting_dims","rhs_contracting_dims","lhs_batch_dims","rhs_batch_dims","lhs_conj","rhs_conj")}
                row["batch_output_policy"] = "native batch-first; axis normalization only during untimed CPU verification"
            if case.get("read_op"):
                row["input_layout_policy"] = (
                    "prepared transposed CUDA view; no input copy inside timing"
                    if backend == "pytorch-cuda" else
                    "same logical values prepared in native JAX representation; no public strided-view equivalence claimed")
            if "_into_accum" in case["op"]:
                row["input_restoration"] = "GPU seed copied into each destination before every batch; synchronized outside timing"
                row["accumulation"] = {"alpha": case.get("alpha", .5), "beta": case.get("beta", 1.)}
            if case["op"] == "dot_general_read_into":
                row["accumulation"] = {"alpha": 1.0, "beta": 0.0}
            unavailable = {**case.get("excluded", {}), **case.get("unavailable", {})}
            numerical_exclusion = backend in case.get("excluded", {}) and backend in case.get("verify_excluded", [])
            if backend in unavailable and (not numerical_exclusion or backend in case.get("unavailable", {})):
                row.update(status="unsupported", reason=unavailable[backend])
                f.write(json.dumps(row) + "\n")
                continue
            case_started = time.monotonic()
            compile_ms = 0.0
            fixture_started = time.monotonic()
            try:
                hosts = [load_array(root, a) for a in case["inputs"]]
                is_fft = case["op"] in {"fft", "ifft", "rfft", "irfft"}
                stored_fingerprints = [load_fingerprint(root, a) for a in case["outputs"]]
                expected = [load_array(root, a) for a in case["outputs"]]
                fixture_ms = (time.monotonic() - fixture_started) * 1e3
                from gpu_multi_output_verification import leaves, verify_outputs, normalized_outputs
                if backend == "pytorch-cuda":
                    inputs = [torch.from_numpy(np.ascontiguousarray(a) if a.ndim else a).to(f"cuda:{device}") for a in hosts]
                    if is_fft:
                        x = inputs[0]
                        operation = getattr(torch.fft, case["op"])
                        op = lambda: operation(x, n=case["n"], dim=-1, norm="backward")
                    else:
                        from gpu_public_api_ops import torch_operation
                        op = torch_operation(torch, case, inputs, verify_only=verify_only)
                    sync = lambda: torch.cuda.synchronize(device)
                    ready = lambda outputs: sync()
                    download = lambda y: y.detach().cpu().numpy()
                    row["synchronization"] = "torch.cuda.synchronize; no D2H in timed region"
                else:
                    inputs = [jax.device_put(a, gpu) for a in hosts]
                    if case.get("jax_ref_addupdate"):
                        from gpu_ref_reuse import prepare_addupdate
                        op = prepare_addupdate(jax, case, inputs, compiled=backend.endswith("-jit"), verify_only=verify_only)
                        row["output_reuse_api"] = "jax.ref.addupdate; caller-owned fixed-buffer Ref"
                    elif is_fft:
                        operation = getattr(jnp.fft, case["op"])
                        fn = lambda value: operation(value, n=case["n"], axis=-1, norm="backward")
                    else:
                        from gpu_public_api_ops import jax_operation, prepare_jax_inputs
                        fn = jax_operation(jax, jnp, case)
                        inputs = prepare_jax_inputs(case, inputs)
                    if not case.get("jax_ref_addupdate"):
                        if backend.endswith("-jit"):
                            compile_started = time.monotonic()
                            fn = jax.jit(fn).lower(*inputs).compile()
                            compile_ms = (time.monotonic() - compile_started) * 1e3
                        op = lambda: fn(*inputs)
                    jax.block_until_ready(inputs)
                    sync = lambda: None  # previous batch waits on every output
                    ready = jax.block_until_ready
                    download = np.asarray
                    row["synchronization"] = "jax.block_until_ready on all retained outputs; no D2H in timed region"
                if hasattr(op, "ready"):
                    ready = op.ready
                    row["synchronization"] = "AOT-compiled scalar token per Ref plus block_until_ready; tokens retained through clock stop; no full snapshot or D2H in timed region"
                reuse = "_into" in case["op"]
                if hasattr(op, "prepare"):
                    op.prepare(1)
                probe = op(0) if reuse else op()
                if hasattr(op, "inspect"):
                    ready([probe])
                    probe = op.inspect(probe)  # snapshot only for untimed verification
                verify_device(probe, backend, device if backend == "pytorch-cuda" else gpu)
                (jax.block_until_ready if backend != "pytorch-cuda" and hasattr(op, "inspect") else ready)([probe])
                verify_started = time.monotonic()
                inspected = normalized_outputs([download(v) for v in leaves(probe)], case, backend)
                # The fingerprint is a recorded diagnostic and never the decision: an
                # output can agree with every stored invariant and still be elementwise
                # wrong, so every element is compared whatever it says.
                fingerprint_compares, fingerprint_disagreements = fingerprint_diagnostic(
                    inspected, stored_fingerprints, case)
                error = verify_outputs(inspected, expected, case, hosts, verify)
                del probe, inspected
                verify_ms = (time.monotonic() - verify_started) * 1e3
                timing = None
                if not verify_only and not numerical_exclusion:
                    timing = measure(op, sync, ready, sum(v.nbytes for v in expected), warmups, runs, reuse=reuse)
                    timing["compile_time_ms"] = compile_ms
                    if hasattr(op, "prepare"):
                        op.prepare(timing["operations_per_sample"])
                        sync()
                    probe = op(timing["operations_per_sample"]-1) if reuse else op()
                    if hasattr(op, "inspect"):
                        ready([probe])
                        probe = op.inspect(probe)
                    verify_device(probe, backend, device if backend == "pytorch-cuda" else gpu)
                    (jax.block_until_ready if backend != "pytorch-cuda" and hasattr(op, "inspect") else ready)([probe])
                    verify_started = time.monotonic()
                    inspected = normalized_outputs([download(v) for v in leaves(probe)], case, backend)
                    more_compares, more_disagreements = fingerprint_diagnostic(
                        inspected, stored_fingerprints, case)
                    fingerprint_compares += more_compares
                    fingerprint_disagreements += more_disagreements
                    error = max(error, verify_outputs(inspected, expected, case, hosts, verify))
                    verify_ms += (time.monotonic() - verify_started) * 1e3
                    del inspected
                    del probe
                timed_ms = sum(timing["batch_ms"]) if timing else None
                case_wall_ms = (time.monotonic() - case_started) * 1e3
                untimed_ms = None if timed_ms is None else max(0.0, case_wall_ms - timed_ms)
                warning = overhead_warning(case["id"], timed_ms, case_wall_ms, overhead_budget)
                if warning:
                    print(f"overhead budget: {warning}", file=sys.stderr)
                row["overhead"] = {
                    "timed_ms": timed_ms,
                    "case_wall_ms": case_wall_ms,
                    "untimed_ms": untimed_ms,
                    "fixture_ms": fixture_ms,
                    "verify_ms": verify_ms,
                    "fingerprints_stored": sum(stored is not None for stored in stored_fingerprints),
                    "fingerprint_compares": fingerprint_compares,
                    "fingerprint_disagreements": fingerprint_disagreements,
                    "budget_ratio": overhead_budget,
                    "over_budget": warning is not None,
                    "warning": warning,
                    "scope": "case wall clock outside the timed region: fixtures, H2D, library initialization, compilation, validation, output destruction",
                }
                row.update(status="unsupported" if numerical_exclusion else ("verification_only" if verify_only else "ok"), timing=timing,
                           verification={"max_error_over_tolerance": error, "reference": case["reference"]})
                if numerical_exclusion:
                    row.update(reason=case["excluded"][backend], cuda_implementation_verified=True)
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
