#!/usr/bin/env python3
"""PyTorch MPS, JAX Metal, and Metal memcpy participants for mac-gpu."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PATTERNS = Path("data/instances/gpu_permutation_mac_patterns.json")
OUTPUT = os.environ.get("BENCH_OUTPUT")
DEVICE_NAME = os.environ.get("GPU_BENCH_DEVICE_NAME", "Apple GPU")
TENFERRO_REVISION = os.environ.get("TENFERRO_BENCH_REVISION", "unknown")


def source_strides(pattern: dict) -> list[int]:
    if pattern["src_layout"]["kind"] == "explicit_strides":
        return pattern["src_layout"]["strides"]
    strides = [1]
    for extent in pattern["shape"][:-1]:
        strides.append(strides[-1] * extent)
    return strides


def logical_source(pattern: dict, flat: np.ndarray) -> np.ndarray:
    return np.lib.stride_tricks.as_strided(
        flat,
        shape=pattern["shape"],
        strides=tuple(stride * flat.itemsize for stride in source_strides(pattern)),
    )


def expected_array(pattern: dict, flat: np.ndarray) -> np.ndarray:
    return np.transpose(logical_source(pattern, flat), axes=pattern["perm"])


def unavailable_record(pattern: dict, backend: str, note: str, status="not_configured") -> dict:
    return base_record(pattern, backend) | {
        "status": status,
        "correctness": "not_run",
        "median_ms": None,
        "p25_ms": None,
        "p75_ms": None,
        "bandwidth_gbs": None,
        "notes": note,
    }


def base_record(pattern: dict, backend: str) -> dict:
    elems = int(np.prod(pattern["shape"], dtype=np.int64))
    return {
        "schema_version": 1,
        "suite_id": "gpu/permutation",
        "target_profile": "mac-gpu",
        "runner": "python",
        "runtime": "Metal",
        "synchronization": "explicit backend completion after each dispatch",
        "allocation": "backend-specific; recorded in notes",
        "tenferro_revision": TENFERRO_REVISION,
        "pattern_id": pattern["id"],
        "label": pattern["label"],
        "backend": backend,
        "shape": pattern["shape"],
        "perm": pattern["perm"],
        "dtype": "f32",
        "elems": elems,
        "bytes_rw": elems * 8,
        "device": DEVICE_NAME,
        "per_call_allocation": backend != "memcpy-metal-d2d",
        "warmup": int(os.environ.get("BENCH_WARMUPS", "3")),
        "iters": max(1, int(os.environ.get("BENCH_RUNS", "7"))),
    }


def timed(record: dict, operation, synchronize) -> tuple[float, float, float, float]:
    for _ in range(record["warmup"]):
        operation()
        synchronize()
    samples = []
    for _ in range(record["iters"]):
        start = time.perf_counter()
        operation()
        synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    median = statistics.median(samples)
    return (
        median,
        float(np.quantile(samples, 0.25)),
        float(np.quantile(samples, 0.75)),
        record["bytes_rw"] / (median * 1e6),
    )


def ok_record(record: dict, values: tuple[float, float, float, float], notes: str) -> dict:
    median, p25, p75, bandwidth = values
    return record | {
        "status": "ok",
        "correctness": "passed",
        "median_ms": median,
        "p25_ms": p25,
        "p75_ms": p75,
        "bandwidth_gbs": bandwidth,
        "notes": notes,
    }


def run_pytorch(pattern: dict, backend: str) -> dict:
    record = base_record(pattern, backend)
    try:
        import torch
    except ImportError as error:
        return unavailable_record(pattern, backend, f"PyTorch import failed: {error}")
    if not torch.backends.mps.is_available():
        return unavailable_record(pattern, backend, "torch.backends.mps.is_available() is false")
    if backend == "pytorch-mps" and len(pattern["shape"]) > 16:
        return unavailable_record(
            pattern,
            backend,
            "PyTorch MPS supports at most 16 dimensions",
            "skipped",
        )

    total = record["elems"]
    host = (np.arange(total, dtype=np.uint32) % 65521).astype(np.float32)
    source = torch.from_numpy(host).to("mps")
    if backend == "memcpy-metal-d2d":
        destination = torch.empty_like(source)
        destination.copy_(source)
        torch.mps.synchronize()
        if not np.array_equal(destination.cpu().numpy(), host):
            return unavailable_record(pattern, backend, "device-copy correctness mismatch", "verification_failed")
        values = timed(record, lambda: destination.copy_(source), torch.mps.synchronize)
        return ok_record(record, values, "destination buffer allocated once and reused")

    view = torch.as_strided(source, pattern["shape"], source_strides(pattern))
    permuted = view.permute(*pattern["perm"])
    out_shape = [pattern["shape"][axis] for axis in pattern["perm"]]

    def allocate_destination():
        storage = torch.empty(total, dtype=torch.float32, device="mps")
        return torch.as_strided(
            storage,
            out_shape,
            source_strides({"shape": out_shape, "src_layout": {"kind": "col_major"}}),
        )

    destination = allocate_destination()
    destination.copy_(permuted)
    torch.mps.synchronize()
    if not np.array_equal(destination.cpu().numpy(), expected_array(pattern, host)):
        return unavailable_record(pattern, backend, "PyTorch MPS correctness mismatch", "verification_failed")

    holder = [destination]

    def operation() -> None:
        destination = allocate_destination()
        destination.copy_(permuted)
        holder[0] = destination

    values = timed(record, operation, torch.mps.synchronize)
    return ok_record(record, values, "fresh compact column-major destination per timed call")


def run_jax(pattern: dict) -> dict:
    backend = "jax-metal"
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        return unavailable_record(pattern, backend, f"JAX import failed: {error}")
    devices = jax.devices()
    platform = any("metal" in device.platform.lower() for device in devices)
    if not platform:
        return unavailable_record(
            pattern,
            backend,
            "no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile",
        )
    record = base_record(pattern, backend)
    total = record["elems"]
    flat = (np.arange(total, dtype=np.uint32) % 65521).astype(np.float32)
    source = jax.device_put(np.array(logical_source(pattern, flat), copy=True, order="C"), devices[0])
    operation = jax.jit(lambda value: jnp.transpose(value, axes=pattern["perm"]))
    actual = operation(source)
    actual.block_until_ready()
    if not np.array_equal(np.asarray(actual), expected_array(pattern, flat)):
        return unavailable_record(pattern, backend, "JAX Metal correctness mismatch", "verification_failed")
    holder = [actual]

    def invoke() -> None:
        holder[0] = operation(source)

    values = timed(record, invoke, lambda: holder[0].block_until_ready())
    return ok_record(record, values, "JIT compilation excluded; logical transpose output")


def emit(record: dict, stream) -> None:
    line = json.dumps(record, separators=(",", ":"))
    print(line)
    if stream is not None:
        stream.write(line + "\n")
        stream.flush()


def main() -> int:
    selected = sys.argv[1] if len(sys.argv) > 1 else "all"
    allowed = {"all", "pytorch-mps", "jax-metal", "memcpy-metal-d2d"}
    if selected not in allowed:
        raise SystemExit(f"unknown backend {selected!r}; choose one of {sorted(allowed)}")
    document = json.loads(PATTERNS.read_text())
    pattern_filter = os.environ.get("PATTERN_ID")
    stream = open(OUTPUT, "a", encoding="utf-8") if OUTPUT else None
    failed = False
    try:
        for pattern in document["patterns"]:
            if pattern_filter and pattern["id"] != pattern_filter:
                continue
            participants = pattern["participants_gpu"]
            for backend in ("pytorch-mps", "jax-metal", "memcpy-metal-d2d"):
                if selected not in ("all", backend) or backend not in participants:
                    continue
                try:
                    record = run_jax(pattern) if backend == "jax-metal" else run_pytorch(pattern, backend)
                except Exception as error:  # preserve the rest of a baseline run
                    record = unavailable_record(pattern, backend, f"{type(error).__name__}: {error}", "runtime_failed")
                emit(record, stream)
                failed |= record["status"] == "verification_failed"
    finally:
        if stream is not None:
            stream.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
