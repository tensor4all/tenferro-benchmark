#!/usr/bin/env python3
"""Python einsum benchmark using PyTorch or JAX backends.

Reads instance JSON files from data/instances/ and benchmarks einsum
with precomputed contraction paths (opt_flops / opt_size).

Output format mirrors the Rust benchmark logs so format_results.py can
parse both Rust and Python results in a unified table.

Usage:
    uv run python scripts/benchmark_python.py --backend pytorch [--num-threads N]
    uv run python scripts/benchmark_python.py --backend jax [--num-threads N]

Environment:
    BENCH_INSTANCE        Run only the named instance (default: all in suite)
    BENCH_SUITE_INCLUDE    Comma-separated instance IDs from the suite YAML
    BENCH_RUNS            Timed runs per instance (default: 15)
    BENCH_WARMUPS         Warmup runs per instance (default: 3)
    OMP_NUM_THREADS       Thread count (overridden by --num-threads)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "instances"

NUM_WARMUP_DEFAULT = 3
NUM_RUNS_DEFAULT = 15
_PYTORCH_THREADS_CONFIGURED = False


def bench_warmups() -> int:
    return int(os.environ.get("BENCH_WARMUPS", NUM_WARMUP_DEFAULT))


def bench_runs() -> int:
    return int(os.environ.get("BENCH_RUNS", NUM_RUNS_DEFAULT))

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OMP_DYNAMIC",
    "RAYON_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "VECLIB_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "XLA_FLAGS",
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python einsum benchmark")
    parser.add_argument(
        "--backend", choices=["pytorch", "jax"], default="pytorch",
        help="Backend to use (default: pytorch)",
    )
    parser.add_argument(
        "--num-threads", type=int,
        default=int(os.environ.get("OMP_NUM_THREADS", "1")),
        help="Number of CPU threads (default: OMP_NUM_THREADS or 1)",
    )
    parser.add_argument(
        "--instance", default=os.environ.get("BENCH_INSTANCE", ""),
        help="Run only this instance name (default: all)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_instances(instance_filter: str = "", suite_include: str = "") -> list[dict]:
    if not suite_include:
        suite_include = os.environ.get("BENCH_SUITE_INCLUDE", "")
    allowed = (
        {name.strip() for name in suite_include.split(",") if name.strip()}
        if suite_include
        else None
    )
    paths = sorted(DATA_DIR.glob("*.json"))
    instances = []
    for path in paths:
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as e:
            print(f"Warning: skip {path.name} ({e})", file=sys.stderr)
            continue
        # data/instances/ also holds non-einsum pattern files (e.g.
        # permutation_patterns.json for cpu/permutation) that are valid JSON
        # but not a single-instance record; skip anything missing "name"
        # instead of crashing, mirroring how the Rust einsum loader treats a
        # missing required field as a parse failure and skips it.
        if not isinstance(d, dict) or "name" not in d:
            print(f"Warning: skip {path.name} (missing field 'name')", file=sys.stderr)
            continue
        if instance_filter:
            if d["name"] != instance_filter:
                continue
        elif allowed is not None and d["name"] not in allowed:
            continue
        instances.append(d)
    return instances


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_format_string(instance: dict) -> str:
    """Return the row-major format string for the instance.

    Prefers ``format_string_rowmajor`` when non-empty; falls back to
    ``format_string`` (which is always row-major for our dataset).
    """
    rowmajor = instance.get("format_string_rowmajor", "")
    return rowmajor if rowmajor else instance["format_string"]


def path_to_opt_einsum(path_data: list[list[int]]) -> list[tuple[int, int]]:
    """Convert JSON path ``[[i,j], ...]`` to opt_einsum format ``[(i,j), ...]``."""
    return [tuple(p) for p in path_data]


def compute_stats(times_ms: list[float]) -> tuple[float, float]:
    """Return (median, IQR) from a sorted list of times in ms."""
    times_ms = sorted(times_ms)
    n = len(times_ms)
    median = times_ms[n // 2]
    q1 = times_ms[n // 4]
    q3 = times_ms[3 * n // 4]
    iqr = q3 - q1
    return median, iqr


def strategy_cache_key(instance: dict, path_meta: dict) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Return a key for identical physical contractions within one backend run."""
    return (
        instance["name"],
        tuple(tuple(pair) for pair in path_meta["path"]),
    )


def configure_thread_env(num_threads: int) -> None:
    value = str(num_threads)
    xla_multi_thread = "true" if num_threads > 1 else "false"
    os.environ.update(
        {
            "OMP_NUM_THREADS": value,
            "OMP_THREAD_LIMIT": value,
            "OMP_DYNAMIC": "FALSE",
            "RAYON_NUM_THREADS": value,
            "OPENBLAS_NUM_THREADS": value,
            "GOTO_NUM_THREADS": value,
            "MKL_NUM_THREADS": value,
            "VECLIB_MAXIMUM_THREADS": value,
            "VECLIB_NUM_THREADS": value,
            "NUMEXPR_NUM_THREADS": value,
            "BLIS_NUM_THREADS": value,
            "XLA_FLAGS": (
                f"--xla_cpu_multi_thread_eigen={xla_multi_thread} "
                f"intra_op_parallelism_threads={value}"
            ),
        }
    )


def configure_pytorch_threads(num_threads: int):
    import torch

    global _PYTORCH_THREADS_CONFIGURED
    if not _PYTORCH_THREADS_CONFIGURED:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        _PYTORCH_THREADS_CONFIGURED = True
    return torch


def xla_backend_name(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return f"xla_{normalized}" if normalized else "xla_unknown"


# ---------------------------------------------------------------------------
# PyTorch backend
# ---------------------------------------------------------------------------

def benchmark_pytorch(
    instance: dict,
    strategy: str,
    num_threads: int,
) -> tuple[tuple[float, float] | None, str | None]:
    """Benchmark one instance with PyTorch.

    Returns ((median_ms, iqr_ms), None) on success, or (None, error_msg).
    """
    import opt_einsum as oe

    torch = configure_pytorch_threads(num_threads)

    dtype_str = instance.get("dtype", "float64")
    if "complex" in dtype_str:
        return None, f"complex dtype ({dtype_str}) not supported"

    fmt = get_format_string(instance)
    shapes = instance["shapes"]
    path = path_to_opt_einsum(instance["paths"][strategy]["path"])

    operands = [torch.zeros(shape, dtype=torch.float64) for shape in shapes]

    try:
        # Warmup
        for _ in range(bench_warmups()):
            oe.contract(fmt, *operands, optimize=path, backend="torch")

        # Timed runs
        times: list[float] = []
        for _ in range(bench_runs()):
            t0 = time.perf_counter()
            oe.contract(fmt, *operands, optimize=path, backend="torch")
            times.append((time.perf_counter() - t0) * 1000.0)

        return compute_stats(times), None

    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


# ---------------------------------------------------------------------------
# JAX backend
# ---------------------------------------------------------------------------

def benchmark_jax(
    instance: dict,
    strategy: str,
) -> tuple[tuple[float, float] | None, str | None]:
    """Benchmark one instance with JAX.

    Returns ((median_ms, iqr_ms), None) on success, or (None, error_msg).
    JAX uses 64-bit floats (requires jax_enable_x64=True).
    block_until_ready() ensures device execution is complete before timing.
    """
    import jax
    import jax.numpy as jnp
    import opt_einsum as oe

    jax.config.update("jax_enable_x64", True)

    dtype_str = instance.get("dtype", "float64")
    if "complex" in dtype_str:
        return None, f"complex dtype ({dtype_str}) not supported"

    fmt = get_format_string(instance)
    shapes = instance["shapes"]
    path = path_to_opt_einsum(instance["paths"][strategy]["path"])

    operands = [jnp.zeros(shape, dtype=jnp.float64) for shape in shapes]

    try:
        # Warmup (first call includes JIT compilation)
        for _ in range(bench_warmups()):
            jax.block_until_ready(
                oe.contract(fmt, *operands, optimize=path, backend="jax")
            )

        # Timed runs
        times: list[float] = []
        for _ in range(bench_runs()):
            t0 = time.perf_counter()
            jax.block_until_ready(
                oe.contract(fmt, *operands, optimize=path, backend="jax")
            )
            times.append((time.perf_counter() - t0) * 1000.0)

        return compute_stats(times), None

    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    backend_name = f"{args.backend}-cpu"
    num_threads = args.num_threads
    configure_thread_env(num_threads)

    instances = load_instances(args.instance)
    if args.instance and not instances:
        print(
            f"BENCH_INSTANCE={args.instance!r}: no matching instance found",
            file=sys.stderr,
        )
        sys.exit(1)
    if not instances:
        print("No benchmark instances matched the suite selection", file=sys.stderr)
        sys.exit(1)

    # ---- Header ----
    print(f"{backend_name} einsum benchmark suite")
    print("==================================")
    print(f"Loaded {len(instances)} instances from {DATA_DIR}")
    print(f"Backend: {backend_name}")
    for key in THREAD_ENV_KEYS:
        print(f"{key}={os.environ.get(key, '')}")
    if args.backend == "pytorch":
        torch = configure_pytorch_threads(num_threads)
        print(f"TORCH_NUM_THREADS={torch.get_num_threads()}")
        print(f"TORCH_NUM_INTEROP_THREADS={torch.get_num_interop_threads()}")
    else:
        import jax

        jax_backend = jax.default_backend()
        print(f"JAX_DEFAULT_BACKEND={jax_backend}")
        print(f"JAX_DOT_BACKEND={xla_backend_name(jax_backend)}")
    print(
        f"Timing: median ± IQR of {bench_runs()} runs "
        f"({bench_warmups()} warmup), path precomputed"
    )

    strategies = ["opt_flops", "opt_size"]
    col_w = 106  # separator width (no Compile column)
    measured_by_path: dict[
        tuple[str, tuple[tuple[int, int], ...]],
        tuple[tuple[float, float] | None, str | None],
    ] = {}

    for strategy in strategies:
        print()
        print(f"Strategy: {strategy}")
        print(
            f"{'Instance':<50} {'Tensors':>8} {'log10FLOPS':>10} "
            f"{'log2SIZE':>12} {'Median (ms)':>12} {'IQR (ms)':>10}"
        )
        print("-" * col_w)

        for idx, instance in enumerate(instances):
            name = instance["name"]
            path_meta = instance["paths"][strategy]
            num_tensors = instance["num_tensors"]
            log10_flops = path_meta["log10_flops"]
            log2_size = path_meta["log2_size"]

            print(
                f"  [{idx + 1}/{len(instances)}] {name}...",
                file=sys.stderr,
            )

            cache_key = strategy_cache_key(instance, path_meta)
            cached = measured_by_path.get(cache_key)
            if cached is not None:
                print(
                    f"  -> {name} strategy={strategy}: "
                    "reusing previous measurement for identical path",
                    file=sys.stderr,
                )
                result, err = cached
            else:
                if args.backend == "pytorch":
                    result, err = benchmark_pytorch(instance, strategy, num_threads)
                else:
                    result, err = benchmark_jax(instance, strategy)
                measured_by_path[cache_key] = (result, err)

            if result is None:
                print(f"  -> {name} (error: {err})", file=sys.stderr)
                print(
                    f"{name:<50} {num_tensors:>8} {log10_flops:>10.2f} "
                    f"{log2_size:>12.2f} {'SKIP':>12} {'-':>10}"
                )
            else:
                median, iqr = result
                print(
                    f"{name:<50} {num_tensors:>8} {log10_flops:>10.2f} "
                    f"{log2_size:>12.2f} {median:>12.3f} {iqr:>10.3f}"
                )


if __name__ == "__main__":
    main()
