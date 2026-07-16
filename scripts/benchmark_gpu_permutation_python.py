#!/usr/bin/env python3
"""GPU permutation / materialize-kernel benchmark suite (gpu/permutation),
Python backends: `pytorch-cuda`, `jax-cuda`, `memcpy-d2d`.

Ports the CUDA-side of `docs/gpu-permutation-suite.md` for the two ML
framework backends and the device-to-device memcpy baseline. The Rust
runner (`src/bin/benchmark_gpu_permutation.rs`) covers
`tenferro-cuda-transpose` / `tenferro-cuda-to-contiguous` / `cutensor`;
this script is the Python counterpart, following the argument/env/sync
conventions of `scripts/benchmark_gpu_python.py`:

- `torch.cuda.synchronize()` after every dispatch (warmup and timed), never
  inside a timed region without also being the timing boundary.
- `jax.block_until_ready()` after every dispatch for the same reason.
- Output downloads for correctness verification happen once, after the
  warmup runs and before any timed loop -- never inside a timed closure
  (AGENTS.md "GPU Timing Fairness").

Column-major destination layout mapping (PyTorch): PyTorch tensors are
always row-major (C-contiguous) at the storage level. A compact
column-major tensor of logical shape `out_shape` has exactly the same
physical byte layout as a C-contiguous tensor of shape
`reversed(out_shape)` with its axes fully reversed. So the destination is
allocated once as `torch.empty(reversed(out_shape), ...)` (a real,
C-contiguous allocation) and a metadata-only `.permute(*reversed_axes)`
view over it is what every timed `copy_` targets; the buffer itself is
allocated once per pattern, outside the timed region, and reused across
every timed iteration.

Environment variables:
- `PATTERN_ID`: run a single pattern id instead of the full suite.
- `BENCH_RUNS` / `BENCH_WARMUPS`: override the size-scaled iteration and
  warmup counts for every participant.
- `BENCH_OUTPUT`: path to write machine-readable JSON Lines results. When
  unset, only the human-readable table is printed to stdout.
- `GPU_BENCH_DEVICE`: CUDA device ordinal (default `0`).
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
PATTERN_PATH = PROJECT_DIR / "data" / "instances" / "permutation_patterns.json"
SUITE_ID = "gpu/permutation"

ALL_BACKENDS = ("pytorch-cuda", "jax-cuda", "memcpy-d2d")


# ---------------------------------------------------------------------------
# Pattern loading (mirrors src/bin/benchmark_gpu_permutation.rs)
# ---------------------------------------------------------------------------

def load_pattern_suite() -> dict[str, Any]:
    suite = json.loads(PATTERN_PATH.read_text())
    if suite.get("version") != 1:
        raise ValueError(f"unsupported pattern schema version {suite.get('version')}")
    if suite.get("index_base") != 0:
        raise ValueError("permute patterns must use index_base = 0")
    if suite.get("semantics") != "out[i0,...,ik] = src[i_perm0,...,i_permk]":
        raise ValueError(f"unsupported semantics {suite.get('semantics')!r}")
    if suite.get("data") != "deterministic_index_value":
        raise ValueError(f"unsupported data mode {suite.get('data')!r}")
    return suite


def col_major_strides(shape: list[int]) -> list[int]:
    strides = [1] * len(shape)
    for i in range(1, len(shape)):
        strides[i] = strides[i - 1] * shape[i - 1]
    return strides


def layout_strides(shape: list[int], layout: dict[str, Any]) -> list[int]:
    if layout["kind"] == "col_major":
        return col_major_strides(shape)
    return list(layout["strides"])


def deterministic_data(total: int) -> np.ndarray:
    return (np.arange(total, dtype=np.float64) + 1.0)


def compute_reference(pattern: dict[str, Any]) -> np.ndarray:
    """Host-computed reference: logical `out_shape` array, dtype float64.

    Only the logical values (not physical byte layout) are compared against
    every backend's output, so this uses `numpy.lib.stride_tricks.as_strided`
    over the deterministic flat buffer for an exact, fast reference -- the
    same source-stride semantics as
    `src/bin/benchmark_gpu_permutation.rs::naive_strided_copy`.
    """
    shape = list(pattern["shape"])
    perm = list(pattern["perm"])
    total = 1
    for d in shape:
        total *= d
    data = deterministic_data(total)
    src_strides = layout_strides(shape, pattern["src_layout"])
    byte_strides = tuple(s * data.itemsize for s in src_strides)
    src_view = np.lib.stride_tricks.as_strided(data, shape=tuple(shape), strides=byte_strides)
    return np.transpose(src_view, axes=perm).astype(np.float64, copy=True)


# ---------------------------------------------------------------------------
# Timing (mirrors src/bin/benchmark_gpu_permutation.rs::bench_n)
# ---------------------------------------------------------------------------

def timing_counts(total: int) -> tuple[int, int]:
    if total >= (1 << 23):
        return 3, 15
    return 5, 40


def timing_counts_env(total: int) -> tuple[int, int]:
    default_warmup, default_iters = timing_counts(total)
    warmup = int(os.environ.get("BENCH_WARMUPS", default_warmup))
    iters = int(os.environ.get("BENCH_RUNS", default_iters))
    return warmup, max(iters, 1)


def bench_n(warmup: int, iters: int, bytes_rw: int, fn) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples_s: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples_s.append(time.perf_counter() - t0)
    samples_s.sort()
    n = len(samples_s)
    if n % 2 == 1:
        median_s = samples_s[n // 2]
    else:
        median_s = (samples_s[n // 2 - 1] + samples_s[n // 2]) / 2.0
    p25_s = samples_s[n // 4]
    p75_s = samples_s[n * 3 // 4]
    return {
        "warmup": warmup,
        "iters": iters,
        "median_ms": median_s * 1e3,
        "p25_ms": p25_s * 1e3,
        "p75_ms": p75_s * 1e3,
        "bandwidth_gbs": (bytes_rw / median_s / 1e9) if median_s > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Result records (mirrors src/bin/benchmark_gpu_permutation.rs::ResultRecord)
# ---------------------------------------------------------------------------

def base_record(
    pattern: dict[str, Any],
    backend: str,
    elems: int,
    bytes_rw: int,
    device: str,
    status: str,
    correctness: str,
    per_call_allocation: bool,
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite_id": SUITE_ID,
        "runner": "python",
        "pattern_id": pattern["id"],
        "label": pattern["label"],
        "backend": backend,
        "shape": list(pattern["shape"]),
        "perm": list(pattern["perm"]),
        "dtype": "f64",
        "elems": elems,
        "bytes_rw": bytes_rw,
        "device": device,
        "status": status,
        "correctness": correctness,
        "per_call_allocation": per_call_allocation,
        "warmup": None,
        "iters": None,
        "median_ms": None,
        "p25_ms": None,
        "p75_ms": None,
        "bandwidth_gbs": None,
        "notes": notes,
    }


def print_human_row(backend: str, timing: dict[str, float] | None, note: str | None) -> None:
    if timing is not None:
        print(
            f"  {backend:30} {timing['median_ms']:8.3f} ms  "
            f"({timing['p25_ms']:.3f} / {timing['p75_ms']:.3f})  "
            f"{timing['bandwidth_gbs']:6.2f} GB/s"
        )
    else:
        print(f"  {backend:30} skipped: {note or 'no timing'}")


def finish_ok(record: dict[str, Any], timing: dict[str, float]) -> dict[str, Any]:
    record.update(timing)
    print_human_row(record["backend"], timing, None)
    return record


def finish_skip(record: dict[str, Any]) -> dict[str, Any]:
    print_human_row(record["backend"], None, record["notes"])
    return record


def verify_output(actual: np.ndarray, reference: np.ndarray) -> str | None:
    if actual.shape != reference.shape:
        return f"shape mismatch {actual.shape} != {reference.shape}"
    if not np.array_equal(actual, reference):
        mismatches = np.argwhere(actual != reference)
        first = tuple(int(i) for i in mismatches[0])
        return f"mismatch at index {first}: {actual[first]} != {reference[first]}"
    return None


# ---------------------------------------------------------------------------
# PyTorch backend
# ---------------------------------------------------------------------------

def run_pytorch(pattern, reference, warmup, iters, bytes_rw, device_name, device_ordinal):
    import torch

    name = "pytorch-cuda"
    device = torch.device(f"cuda:{device_ordinal}")
    shape = list(pattern["shape"])
    perm = list(pattern["perm"])
    out_shape = [shape[p] for p in perm]
    total = int(np.prod(shape)) if shape else 1

    src_strides = layout_strides(shape, pattern["src_layout"])
    flat = torch.from_numpy(deterministic_data(total)).to(device)
    src_view = torch.as_strided(flat, size=shape, stride=src_strides)
    src_permuted = src_view.permute(*perm)

    rank = len(shape)
    reversed_axes = list(range(rank))[::-1]
    dst_native = torch.empty([out_shape[a] for a in reversed_axes], dtype=torch.float64, device=device)
    dst_view = dst_native.permute(*reversed_axes)
    assert list(dst_view.shape) == out_shape, "column-major destination view shape must match out_shape"

    def op():
        dst_view.copy_(src_permuted)
        torch.cuda.synchronize(device)

    op()  # correctness run (warmup #1), not timed
    actual = dst_view.detach().to("cpu", torch.float64).numpy()
    err = verify_output(actual, reference)
    if err is not None:
        return finish_skip(
            base_record(pattern, name, total, bytes_rw, device_name, "verification_failed", "failed", False, err)
        )

    timing = bench_n(warmup, iters, bytes_rw, op)
    return finish_ok(
        base_record(pattern, name, total, bytes_rw, device_name, "ok", "passed", False),
        timing,
    )


# ---------------------------------------------------------------------------
# JAX backend
# ---------------------------------------------------------------------------

def run_jax(pattern, reference, warmup, iters, bytes_rw, device_name, device_ordinal):
    import jax
    import jax.numpy as jnp

    # MANDATORY: jax defaults to f32 silently otherwise.
    jax.config.update("jax_enable_x64", True)

    name = "jax-cuda"
    shape = list(pattern["shape"])
    perm = list(pattern["perm"])
    total = int(np.prod(shape)) if shape else 1

    if pattern["src_layout"]["kind"] != "col_major":
        return finish_skip(
            base_record(
                pattern, name, total, bytes_rw, device_name, "skipped", "skipped", True,
                "jax cannot express arbitrary source strides through jnp's public API",
            )
        )

    try:
        cuda_devices = jax.devices("cuda")
    except RuntimeError as exc:
        return finish_skip(
            base_record(pattern, name, total, bytes_rw, device_name, "not_configured", "skipped", True, str(exc)[:300])
        )
    device = cuda_devices[device_ordinal] if device_ordinal < len(cuda_devices) else cuda_devices[0]

    data = deterministic_data(total)
    src_np = data.reshape(shape, order="F")
    src_jax = jax.device_put(jnp.asarray(src_np), device)

    # Rank-24 XLA compile hazard: jit compilation of a rank-24 transpose was
    # observed to run >40 min at 100% host CPU without completing (jax 0.10.1,
    # RTX 3060), so tn_light_415_24d_contiguous_same_perm excludes jax-cuda
    # via its participants_gpu gate; this code path is never reached for it.
    perm_fn = jax.jit(lambda x: jnp.transpose(x, perm))

    def op():
        out = perm_fn(src_jax)
        out.block_until_ready()
        return out

    out = op()  # correctness run (warmup #1), not timed
    actual = np.asarray(out)
    err = verify_output(actual, reference)
    if err is not None:
        return finish_skip(
            base_record(pattern, name, total, bytes_rw, device_name, "verification_failed", "failed", True, err)
        )

    timing = bench_n(warmup, iters, bytes_rw, op)
    return finish_ok(
        base_record(
            pattern, name, total, bytes_rw, device_name, "ok", "passed", True,
            "jax allocates a fresh output array per call (functional transpose); destination is not reused. "
            "Output is materialized in XLA's default row-major layout (col-major not requestable via the "
            "public API), so at the byte level this is the reversal-conjugate permutation task -- equivalent "
            "difficulty class, not byte-identical to the col-major columns (see docs/gpu-permutation-suite.md)",
        ),
        timing,
    )


# ---------------------------------------------------------------------------
# memcpy-d2d backend (torch, contiguous baseline only)
# ---------------------------------------------------------------------------

def run_memcpy_d2d(pattern, reference, warmup, iters, bytes_rw, device_name, device_ordinal):
    import torch

    name = "memcpy-d2d"
    device = torch.device(f"cuda:{device_ordinal}")
    shape = list(pattern["shape"])
    total = int(np.prod(shape)) if shape else 1

    identity = list(pattern["perm"]) == list(range(len(shape)))
    contiguous = pattern["src_layout"]["kind"] == "col_major" and pattern["dst_layout"]["kind"] == "col_major"
    if not (identity and contiguous):
        return finish_skip(
            base_record(
                pattern, name, total, bytes_rw, device_name, "skipped", "skipped", False,
                "memcpy-d2d only participates in the contiguous identity-permutation baseline",
            )
        )

    src = torch.from_numpy(deterministic_data(total)).to(device)
    dst = torch.empty_like(src)

    def op():
        dst.copy_(src)
        torch.cuda.synchronize(device)

    op()  # correctness run (warmup #1), not timed
    actual = dst.detach().to("cpu", torch.float64).numpy().reshape(shape, order="F")
    err = verify_output(actual, reference)
    if err is not None:
        return finish_skip(
            base_record(pattern, name, total, bytes_rw, device_name, "verification_failed", "failed", False, err)
        )

    timing = bench_n(warmup, iters, bytes_rw, op)
    return finish_ok(
        base_record(pattern, name, total, bytes_rw, device_name, "ok", "passed", False),
        timing,
    )


_RUNNERS = {
    "pytorch-cuda": run_pytorch,
    "jax-cuda": run_jax,
    "memcpy-d2d": run_memcpy_d2d,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def gpu_name(device_ordinal: int) -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", str(device_ordinal)],
            check=True, capture_output=True, text=True,
        )
        name = proc.stdout.strip()
        if name:
            return name
    except (OSError, subprocess.CalledProcessError):
        pass
    return f"cuda:{device_ordinal}"


def torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def run_pattern(pattern, backends, device_name, device_ordinal, records: list[dict[str, Any]]) -> None:
    reference = compute_reference(pattern)
    total = int(reference.size)
    bytes_rw = total * 8 * 2
    warmup, iters = timing_counts_env(total)

    print(f"=== {pattern['label']} ===\n  id={pattern['id']} elems={total} bytes(r+w)={bytes_rw}")

    participants = set(pattern.get("participants_gpu", []))
    for backend in backends:
        if backend not in participants:
            continue
        runner = _RUNNERS[backend]
        try:
            record = runner(pattern, reference, warmup, iters, bytes_rw, device_name, device_ordinal)
        except Exception as exc:  # noqa: BLE001 - convert unexpected runtime errors into a skip record
            record = finish_skip(
                base_record(pattern, backend, total, bytes_rw, device_name, "runtime_failed", "skipped", False, str(exc)[:500])
            )
        records.append(record)
    print()


def main() -> int:
    backends = [b for b in ALL_BACKENDS if not sys.argv[1:] or b in sys.argv[1:]]

    device_ordinal = int(os.environ.get("GPU_BENCH_DEVICE", "0"))
    pattern_filter = os.environ.get("PATTERN_ID")
    bench_output = os.environ.get("BENCH_OUTPUT")

    print("tenferro-benchmark GPU permutation suite (gpu/permutation, python)")
    print("====================================================================")
    print(f"Backends: {', '.join(backends)}")
    print(f"Patterns: {PATTERN_PATH}")
    if pattern_filter:
        print(f"Pattern filter: {pattern_filter}")
    if bench_output:
        print(f"JSON output: {bench_output}")

    if not torch_cuda_available():
        print("benchmark_gpu_permutation_python: no CUDA GPU found (torch.cuda.is_available()=False), skipping",
              file=sys.stderr)
        if bench_output:
            Path(bench_output).write_text("")
        return 0

    device_name = gpu_name(device_ordinal)
    print(f"Device: cuda:{device_ordinal} ({device_name})")
    print()

    suite = load_pattern_suite()
    patterns = suite["patterns"]
    if pattern_filter:
        patterns = [p for p in patterns if p["id"] == pattern_filter]
        if not patterns:
            print(f"PATTERN_ID={pattern_filter} did not match any pattern in {PATTERN_PATH}", file=sys.stderr)
            return 1

    records: list[dict[str, Any]] = []
    print("--- Correctness verification and benchmarks ---")
    for pattern in patterns:
        run_pattern(pattern, backends, device_name, device_ordinal, records)

    if bench_output:
        with open(bench_output, "w") as fh:
            for record in records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")

    if any(r["status"] == "verification_failed" for r in records):
        print("one or more participants failed the correctness gate", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
