#!/usr/bin/env python3
"""Format GPU linalg JVP/VJP JSONL records as a backend comparison table."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

import report_summary

from collect_cpu_info import collect_cpu_info, markdown as cpu_info_markdown
from collect_gpu_info import markdown as gpu_info_markdown, resolve_gpu_info

BACKEND_ORDER = [
    "tenferro-cuda-trace",
    "tenferro-cuda-trace-deferred-batch",
    "pytorch-cuda",
    "jax-cuda",
]

BACKEND_LABELS = {
    "tenferro-cuda-trace": "tenferro-rs CUDA trace",
    "tenferro-cuda-trace-deferred-batch": "tenferro-rs CUDA trace (deferred batch throughput)",
    "pytorch-cuda": "PyTorch CUDA",
    "jax-cuda": "JAX CUDA JIT",
}

LOSS_NOTES = {
    "grad_sum_svd_s": "loss = sum(singular values); w.r.t. input matrix A",
    "grad_sum_qr": "loss = sum(Q) + sum(R); w.r.t. input matrix A",
    "grad_sum_eigh": "loss = sum(eigenvalues); w.r.t. SPD input matrix A",
    "grad_sum_lu": "loss = sum(L) + sum(U); w.r.t. input matrix A",
    "grad_sum_solve": "loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)",
}


def clean_markdown_eof(markdown: str) -> str:
    return markdown.rstrip() + "\n"


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def load_run_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open() as fh:
        metadata = yaml.safe_load(fh)
    return metadata if isinstance(metadata, dict) else None


def format_cell(record: dict[str, Any] | None) -> str:
    if record is None:
        return "-"
    status = record["status"]
    if status == "ok":
        median = record["timing"]["median_ms"]
        iqr = record["timing"]["iqr_ms"]
        if median is None:
            return "ok (missing median)"
        if iqr is None:
            return f"{median:.3f}"
        return f"{median:.3f} ± {iqr:.3f}"
    if status == "not_configured":
        return "not configured"
    return status.replace("_", " ")


def shape_from_problem_id(problem_id: str, loss: str) -> str:
    parts = problem_id.split("_")
    n = parts[-1]
    if loss == "grad_sum_solve":
        return f"{n}x{n},rhs=1"
    return f"{n}x{n}"


def problem_meta(problem_id: str) -> tuple[str, str, str]:
    # linalg_ad_{suite}_{loss}_{phase}_f64_{n}
    parts = problem_id.split("_")
    suite = parts[2]
    loss = "_".join(parts[3:-3])
    phase = parts[-3]
    benchmark = f"{loss}_{phase}"
    shape = shape_from_problem_id(problem_id, loss)
    return suite, benchmark, shape


def format_markdown(
    records: list[dict[str, Any]],
    run_metadata: dict[str, Any] | None = None,
) -> str:
    seen = set()
    for record in records:
        key = record["problem_id"], record["backend"]
        if key in seen or record["backend"] not in BACKEND_ORDER:
            raise ValueError(f"duplicate/unselected AD result: {key}")
        seen.add(key)
        if record["status"] == "ok":
            verification = record["verification"]
            if verification["status"] != "passed" or not verification.get("reference_backend"):
                raise ValueError(f"unverified AD result: {key}")
            for field in ("max_abs_error", "max_rel_error", "residual"):
                value = verification.get(field)
                if value is None or not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid AD verification: {key}")
            if verification["residual"] > 1:
                raise ValueError(f"AD verification exceeds tolerance: {key}")
            sampling = json.loads(record["execution"]["notes"])["sampling"]
            count, samples = sampling["operations_per_sample"], sampling["batch_ms"]
            if not isinstance(count, int) or count < 1 or len(samples) != record["timing"]["timed_runs"] or not samples:
                raise ValueError(f"invalid AD sampling: {key}")
            if any(not math.isfinite(v) or v <= 0 for v in samples):
                raise ValueError(f"invalid AD batch durations: {key}")
        elif record["status"] != "unsupported":
            raise ValueError(f"cannot publish AD status {record['status']}: {key}")
    if run_metadata and run_metadata.get("suite_file"):
        suite = yaml.safe_load(Path(run_metadata["suite_file"]).read_text())
        expected = {(p["id"], b) for p in suite["problems"] for b in suite["backends"]}
        if seen != expected:
            raise ValueError(f"incomplete/unselected AD matrix: missing={sorted(expected-seen)}, extra={sorted(seen-expected)}")
    by_key: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        suite, benchmark, shape = problem_meta(record["problem_id"])
        key = (suite, benchmark, "f64", shape)
        backend = record["backend"]
        if backend in BACKEND_ORDER:
            by_key[key][backend] = record

    lines = ["# GPU Linalg JVP/VJP Benchmark Results", ""]
    if run_metadata:
        lines.extend(
            [
                f"- Target profile: `{run_metadata.get('target_profile', 'nvidia-gpu')}`",
                f"- Suite: `{run_metadata.get('suite_id', 'gpu/linalg_jvp_vjp')}`",
                f"- Suite file: `{run_metadata.get('suite_file', 'benchmarks/gpu/linalg_jvp_vjp.yaml')}`",
                f"- Timestamp: `{run_metadata.get('timestamp', 'unknown')}`",
            ]
        )
        tenferro = run_metadata.get("tenferro_rs")
        if isinstance(tenferro, dict) and tenferro.get("commit"):
            lines.append(f"- tenferro-rs commit: `{tenferro['commit']}`")
        lines.append("")

    gpu_info = resolve_gpu_info(records, run_metadata)
    gpu_markdown = gpu_info_markdown(gpu_info)
    if gpu_markdown:
        lines.append(gpu_markdown.rstrip())
        lines.append("")
    lines.append(cpu_info_markdown(collect_cpu_info()).rstrip())
    lines.append("")
    lines.append("Median ± IQR (ms). Missing backends are shown as `-`.")
    lines.append("")
    lines.append(
        "tenferro-rs JVP/VJP use trace-mode `AdContext` on CUDA; PyTorch uses "
        "`torch.func.jvp` / `vjp` on CUDA."
    )
    lines.append("JAX uses an explicitly lowered/compiled whole-operation JIT; compilation and lazy initialization occur before sampling. Primal loss and first-order derivatives with respect to A are both returned and checked against CPU F64 AD, whose directional derivative is checked by central differences. RHS and derivative seeds are prepared before timing.")
    lines.append("Standard tenferro trace rows run the prepared graph in one caller-held CUDA session (Runtime::run_prepared_in); each timed call includes the run's own stream-retirement barrier. Small trace rows are separately labelled public-path latency diagnostics including internal session entry. Any unsupported row retains its actual reason in the raw records.")
    if any(
        record["backend"] == "tenferro-cuda-trace-deferred-batch" for record in records
    ):
        lines.append(
            "Rows whose backend name ends in `-deferred-batch` are **throughput mode**: each "
            "timed sample enqueues K deferred prepared runs "
            "(`Runtime::run_prepared_deferred`) in one caller-held CUDA session and pays one "
            "retirement barrier (`finish_prepared`), so the reported per-operation time is "
            "`sample_ms / K`. They measure batched throughput and are not comparable with, "
            "and never averaged together with, the per-call latency rows above."
        )
    lines.append(
        "Inputs are uploaded to the GPU before timed runs; initial host-to-device "
        "transfer is outside the timed region."
    )
    lines.append(
        "Timed runs include the host API call and backend-native device synchronization "
        "without downloading AD outputs in the timed region."
    )
    suites_present = {suite for suite, _, _, _ in by_key}
    if "small" in suites_present:
        lines.append("Small cases (n=2, 4, 8) use bounded batches of repeated declared operations with retained outputs. They are public-path latency diagnostics, not isolated GPU throughput or shared-session comparisons. Native synchronization ends each batch; numerical downloads are outside timing.")
    if suites_present == {"small"}:
        lines.append(
            "This suite exists to measure single-call latency and per-op overhead: the "
            "device kernels are a small fraction of each row, so the numbers are not GPU "
            "throughput. GPU-sized AD results live in `result/nvidia-gpu/gpu/linalg_jvp_vjp.md`."
        )
    lines.append("")

    lines.extend(
        [
            "## Linalg JVP/VJP Benchmark Items",
            "",
            "| suite | benchmark | dtype | shape | "
            + " | ".join(BACKEND_LABELS[b] for b in BACKEND_ORDER)
            + " |",
            "|---|---|---:|---|"
            + "|".join("---:" for _ in BACKEND_ORDER)
            + "|",
        ]
    )

    for suite, benchmark, dtype, shape in sorted(by_key):
        backends = by_key[(suite, benchmark, dtype, shape)]
        row = [
            suite,
            f"`{benchmark}`",
            dtype,
            f"`{shape}`",
            *(format_cell(backends.get(backend)) for backend in BACKEND_ORDER),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Sampling and unsupported routes", "", "| Case | Backend | Ops/sample | Samples | Scope / reason |", "| --- | --- | ---: | ---: | --- |"])
    for record in records:
        if record["status"] == "ok":
            sample = json.loads(record["execution"]["notes"])["sampling"]
            count, samples, scope = sample["operations_per_sample"], len(sample["batch_ms"]), record["timing"]["timing_scope"]
            if record["backend"] == "tenferro-cuda-trace-deferred-batch":
                # Throughput rows declare their own batched scope.
                scope = sample.get("scope", scope)
        else:
            count, samples, scope = "—", "—", record["execution"]["unsupported_reason"]
        lines.append(f"| {record['problem_id']} | {record['backend']} | {count} | {samples} | {scope.replace('|', '/')} |")
    covered = sorted({benchmark.rsplit("_", 1)[0] for _, benchmark, _, _ in by_key})
    if covered:
        lines.extend(["", "## Loss Definitions", ""])
        for op in covered:
            note = LOSS_NOTES.get(op)
            if note:
                lines.append(f"- `{op}`: {note}")

    return clean_markdown_eof("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_summary(
    records: list[dict[str, Any]],
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Machine-readable sidecar for the same records the Markdown is built from."""
    header = report_summary.metadata_from_run_metadata(run_metadata)
    hardware = dict(header["hardware"])
    gpu_info = resolve_gpu_info(records, run_metadata)
    if gpu_info.get("Name"):
        hardware.setdefault("gpu", gpu_info["Name"])

    def group(record: dict[str, Any]) -> str:
        _suite, benchmark, _shape = problem_meta(record["problem_id"])
        return benchmark

    return report_summary.build_document(
        report_summary.rows_from_gpu_records(records, group=group),
        target_profile=header["target_profile"],
        suite_id=header["suite_id"],
        timestamp=header["timestamp"],
        tenferro_commit=header["tenferro_commit"],
        hardware=hardware,
        unit="ms",
    )


def main() -> int:
    args = parse_args()
    records = load_records(args.results)
    run_metadata = load_run_metadata(args.run_metadata)
    markdown = format_markdown(
        records,
        run_metadata,
    )
    summary = build_summary(records, run_metadata)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
        report_summary.write_document(
            report_summary.summary_path_for_report(args.output), summary
        )
    else:
        sys.stdout.write(markdown)
    report_summary.maybe_write(None if args.output else summary, env=os.environ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
