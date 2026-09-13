#!/usr/bin/env python3
"""Format CPU benchmark CSV rows as a backend comparison table."""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import defaultdict
from math import prod
from pathlib import Path
from typing import NamedTuple


def clean_markdown_eof(markdown: str) -> str:
    return markdown.rstrip() + "\n"


BACKEND_ORDER = [
    "tenferro-direct",
    "tenferro-eager",
    "tenferro-trace",
    "pytorch-cpu",
    "jax-cpu",
    "julia-base",
    "strided-jl",
]

BACKEND_LABELS = {
    "tenferro-direct": "tenferro-rs direct API (ms)",
    "tenferro-eager": "tenferro-rs eager mode (ms)",
    "tenferro-trace": "tenferro-rs trace mode (ms)",
    "pytorch-cpu": "PyTorch Python (ms)",
    "jax-cpu": "JAX Python (XLA CPU) (ms)",
    "julia-base": "Julia (Base/LinearAlgebra) (ms)",
    "strided-jl": "Julia (Strided.jl) (ms)",
}

PLAUSIBILITY_RATIO_LIMIT = 10.0
CONSERVATIVE_BANDWIDTH_GBPS_PER_THREAD = 100.0
CONSERVATIVE_BANDWIDTH_GBPS_MAX = 1_000.0
CONSERVATIVE_FLOPS_GFLOPS_PER_THREAD = 50.0
CONSERVATIVE_FLOPS_GFLOPS_MAX = 3_200.0

DTYPE_BYTES = {
    "bool": 1,
    "f32": 4,
    "f64": 8,
    "i32": 4,
    "i64": 8,
    "c32": 8,
    "c64": 16,
    "f64->f32": 8,
}

PUBLIC_API_MATERIALIZING_SUITES = {
    "cpu/complex",
    "cpu/elementwise_reduction",
    "cpu/einsum_concrete",
    "cpu/indexing_layout",
    "cpu/linalg_uncovered",
    "cpu/output_reuse",
    "cpu/structural_shape",
}

OUTPUT_SHAPE_REQUIRED_BENCHMARKS = {
    "broadcast_in_dim",
    "embed_diagonal",
    "extract_diagonal",
    "reshape",
}

REDUCTION_BENCHMARKS = {
    "norm_fro",
    "reduce_max_axis0",
    "reduce_min_axis1",
    "reduce_prod_all",
    "reduce_sum_all",
}

MATRIX_PRODUCT_BENCHMARKS = {
    "dot_general",
    "dot_general_read_into",
    "dot_general_read_into_accum",
    "dot_general_with_conj",
    "einsum_ij_jk_ik",
    "tensordot",
}


class WorkloadShape(NamedTuple):
    primary: tuple[int, ...]
    output: tuple[int, ...]
    rhs: int | None
    has_explicit_output: bool


def row_value(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value != "":
            return value
    return ""


def normalize_backend(row: dict[str, str]) -> str:
    backend = row_value(row, "backend").strip()
    if backend in {"cpu-faer", "system-openblas", "system-accelerate", "system-mkl"}:
        return "tenferro-eager"
    return backend


def normalize_row(row: dict[str, str]) -> tuple[tuple[str, str, str, str, str], str, str]:
    suite = row_value(row, "suite")
    benchmark = row_value(row, "benchmark", "name", "op")
    phase = row_value(row, "phase")
    if phase and phase != "primal":
        benchmark = f"{benchmark}_{phase}"
    dtype = row_value(row, "dtype")
    threads = row_value(row, "threads")
    shape = row_value(row, "shape")
    backend = normalize_backend(row)

    median_ms = row_value(row, "median_ms")
    iqr_ms = row_value(row, "iqr_ms")
    if not median_ms:
        mean_us = row_value(row, "mean_us")
        if mean_us:
            median_ms = f"{float(mean_us) / 1000.0:.6f}"
    if not median_ms:
        status = row_value(row, "status") or "-"
        value = status if status != "ok" else "-"
    elif iqr_ms:
        value = f"{float(median_ms):.3f} ± {float(iqr_ms):.3f}"
    else:
        value = f"{float(median_ms):.3f}"

    return (suite, benchmark, dtype, threads, shape), backend, value


def plausibility_findings(
    timings: dict[tuple[str, str, str, str, str], dict[str, float]],
) -> list[str]:
    findings: list[str] = []
    for key in sorted(timings):
        values = timings[key]
        if len(values) < 2:
            continue
        reference_backend, reference_ms = max(values.items(), key=lambda item: item[1])
        suite, benchmark, dtype, threads, shape = key
        for backend, median_ms in sorted(values.items()):
            ratio = reference_ms / median_ms
            if ratio <= PLAUSIBILITY_RATIO_LIMIT:
                continue
            findings.append(
                f"- `{suite}/{benchmark}` ({dtype}, threads={threads}, shape=`{shape}`): "
                f"`{backend}` is {ratio:.1f}x faster than the slowest successful cell "
                f"(`{reference_backend}`, {reference_ms:.3f} ms). Audit fixture semantics, "
                "synchronization, labels, and operation-specific bandwidth/FLOP bounds."
            )
    return findings


def parse_dense_shape(shape: str) -> tuple[int, ...] | None:
    normalized = shape.strip().lower().replace(" ", "")
    if re.fullmatch(r"\d+(?:\+\d+)+", normalized):
        return (sum(int(value) for value in normalized.split("+")),)
    if not re.fullmatch(r"\d+(?:x\d+)*", normalized):
        return None
    return tuple(int(value) for value in normalized.split("x"))


def parse_workload_shape(shape: str) -> WorkloadShape | None:
    normalized = shape.strip().lower().replace(" ", "")
    left, separator, right = normalized.partition("->")
    primary_text = left.split(",", 1)[0]
    output_text = right.split(",", 1)[0] if separator else primary_text
    primary = parse_dense_shape(primary_text)
    output = parse_dense_shape(output_text)
    if primary is None or output is None:
        return None
    rhs_match = re.search(r"(?:^|,)rhs=(\d+)(?:,|$)", normalized)
    return WorkloadShape(
        primary=primary,
        output=output,
        rhs=int(rhs_match.group(1)) if rhs_match else None,
        has_explicit_output=bool(separator),
    )


def estimated_flops(benchmark: str, workload: WorkloadShape) -> int:
    shape = workload.primary
    if benchmark in REDUCTION_BENCHMARKS:
        return prod(shape)
    if len(shape) < 2:
        return 0
    rows, cols = shape[-2:]
    batch = prod(shape[:-2]) if len(shape) > 2 else 1
    if benchmark in MATRIX_PRODUCT_BENCHMARKS and rows == cols:
        # One scalar multiply-accumulate per inner dimension is a strict lower
        # estimate for real and complex matrix products.
        return batch * rows**3
    if benchmark == "triangular_solve" and rows == cols and workload.rhs:
        return batch * rows * rows * workload.rhs // 2
    if benchmark in {"eig", "eigvals", "eigvalsh"} and rows == cols:
        return batch * rows**3
    if benchmark in {"cholesky", "det", "inv", "lu", "slogdet", "solve"} and rows == cols:
        return batch * rows**3 // 3
    if benchmark in {"lstsq", "pinv", "pinv_with_rtol", "qr", "svd", "svd_full"}:
        return batch * rows * cols * min(rows, cols)
    return 0


def estimated_minimum_bytes(
    suite: str,
    benchmark: str,
    workload: WorkloadShape,
    dtype_bytes: int,
    flops: int,
) -> int:
    if suite == "cpu/view_metadata":
        return 0
    if suite not in PUBLIC_API_MATERIALIZING_SUITES and not flops:
        return 0
    if benchmark in OUTPUT_SHAPE_REQUIRED_BENCHMARKS and not workload.has_explicit_output:
        return 0
    if benchmark in {"slice", "dynamic_slice"}:
        if workload.has_explicit_output:
            logical_elements = prod(workload.output)
        else:
            legacy_output_elements = {
                ("slice", (4_194_304,)): 2_096_128,
                ("dynamic_slice", (4_194_304,)): 2_097_152,
            }.get((benchmark, workload.primary))
            if legacy_output_elements is None:
                return 0
            logical_elements = legacy_output_elements
    elif benchmark in REDUCTION_BENCHMARKS:
        logical_elements = prod(workload.primary)
    else:
        logical_elements = prod(workload.output)
    return logical_elements * dtype_bytes


def physical_bound_findings(
    timings: dict[tuple[str, str, str, str, str], dict[str, float]],
) -> list[str]:
    findings: list[str] = []
    for key in sorted(timings):
        suite, benchmark, dtype, threads_text, shape_text = key
        workload = parse_workload_shape(shape_text)
        dtype_bytes = DTYPE_BYTES.get(dtype)
        if workload is None or dtype_bytes is None:
            continue
        try:
            threads = max(1, int(threads_text))
        except ValueError:
            continue

        flops = estimated_flops(benchmark, workload)
        bytes_moved = estimated_minimum_bytes(
            suite,
            benchmark,
            workload,
            dtype_bytes,
            flops,
        )
        bandwidth_gbps = min(
            CONSERVATIVE_BANDWIDTH_GBPS_PER_THREAD * threads,
            CONSERVATIVE_BANDWIDTH_GBPS_MAX,
        )
        compute_gflops = min(
            CONSERVATIVE_FLOPS_GFLOPS_PER_THREAD * threads,
            CONSERVATIVE_FLOPS_GFLOPS_MAX,
        )
        bandwidth_lower_ms = (
            bytes_moved / (bandwidth_gbps * 1.0e9) * 1.0e3 if bytes_moved else 0.0
        )
        compute_lower_ms = flops / (compute_gflops * 1.0e9) * 1.0e3 if flops else 0.0
        lower_bound_ms = max(bandwidth_lower_ms, compute_lower_ms)
        if lower_bound_ms == 0.0:
            continue

        for backend, median_ms in sorted(timings[key].items()):
            ratio = lower_bound_ms / median_ms
            if ratio <= PLAUSIBILITY_RATIO_LIMIT:
                continue
            observed_parts = []
            if bytes_moved:
                observed_parts.append(f"{bytes_moved / (median_ms * 1.0e6):.1f} GB/s")
            if flops:
                observed_parts.append(f"{flops / (median_ms * 1.0e6):.1f} GFLOP/s")
            work_parts = []
            if bytes_moved:
                work_parts.append(f"{bytes_moved} minimum logical bytes")
            if flops:
                work_parts.append(f"{flops} estimated FLOPs")
            findings.append(
                f"- `{suite}/{benchmark}` ({dtype}, threads={threads}, shape=`{shape_text}`), "
                f"`{backend}`: {', '.join(work_parts)} imply "
                f"{' and '.join(observed_parts)} at {median_ms:.3f} ms. The conservative "
                f"physical lower bound is {lower_bound_ms:.3f} ms, so the cell is "
                f"{ratio:.1f}x faster than the bound."
            )
    return findings


def format_table(paths: list[Path]) -> str:
    by_key: dict[tuple[str, str, str, str, str], dict[str, str]] = defaultdict(dict)
    timings: dict[tuple[str, str, str, str, str], dict[str, float]] = defaultdict(dict)
    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                key, backend, value = normalize_row(row)
                if backend:
                    by_key[key][backend] = value
                    median_ms = row_value(row, "median_ms")
                    status = row_value(row, "status")
                    if median_ms and status == "ok" and float(median_ms) > 0:
                        timings[key][backend] = float(median_ms)

    lines = [
        "## CPU Benchmark Items",
        "",
        "Median ± IQR (ms). Missing backends are shown as `-`.",
        "",
        "| suite | benchmark | dtype | threads | shape | "
        + " | ".join(BACKEND_LABELS[b] for b in BACKEND_ORDER)
        + " |",
        "|---|---|---:|---:|---|"
        + "|".join("---:" for _ in BACKEND_ORDER)
        + "|",
    ]

    for key in sorted(by_key):
        suite, benchmark, dtype, threads, shape = key
        values = by_key[key]
        row = [
            suite,
            f"`{benchmark}`",
            dtype,
            threads,
            f"`{shape}`",
            *(values.get(backend, "-") for backend in BACKEND_ORDER),
        ]
        lines.append("| " + " | ".join(row) + " |")

    spread_findings = plausibility_findings(timings)
    lines.extend(
        [
            "",
            "## Cross-Backend Spread Audit",
            "",
            (
                "Rows with a successful cell more than 10x faster than the slowest "
                "successful cell are flagged for operation-specific review. This "
                "cross-backend spread check is a warning, not a correctness verdict."
            ),
            "",
        ]
    )
    if spread_findings:
        lines.extend(spread_findings)
    else:
        lines.append("No cross-backend spread above 10x was detected.")

    physical_findings = physical_bound_findings(timings)
    lines.extend(
        [
            "",
            "## Physical Bound Audit",
            "",
            (
                "This independent check derives minimum logical bytes from dtype and shape "
                "for materializing public API operations, plus conservative operation FLOPs "
                "for reductions, Frobenius norms, matrix products, triangular solves, and "
                "recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) "
                "are parsed explicitly. Operations whose touched region cannot be inferred "
                "conservatively from the reported shape are left unmodeled. "
                "It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped "
                "at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting "
                "lower bound is flagged. These deliberately generous ceilings are a sanity "
                "screen, not a performance gate."
            ),
            "",
        ]
    )
    if physical_findings:
        lines.extend(physical_findings)
    else:
        lines.append("No cell exceeded the conservative physical bound by more than 10x.")

    return clean_markdown_eof("\n".join(lines))


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cpu-ops.csv>...", file=sys.stderr)
        sys.exit(1)
    sys.stdout.write(format_table([Path(value) for value in sys.argv[1:]]))


if __name__ == "__main__":
    main()
