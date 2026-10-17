"""Parse benchmark log files and format results as a markdown table.

Usage:
    python scripts/format_results.py data/results/<target_profile>/cpu/einsum/<timestamp>/tenferro_trace_*.log data/results/<target_profile>/cpu/einsum/<timestamp>/tenferro_eager_*.log
    python scripts/format_results.py data/results/<target_profile>/cpu/einsum/<timestamp>/*.log   # all backends

Supports log formats produced by:
  - tenferro trace/eager (Rust) — columns: Instance Tensors log10FLOPS log2SIZE Median IQR Compile
  - strided-opteinsum (Rust)
  - pytorch-cpu / jax-cpu — columns: Instance Tensors log10FLOPS log2SIZE Median IQR
"""

import argparse
import os
import re
import sys
from pathlib import Path

import report_summary


def clean_markdown_eof(markdown: str) -> str:
    return markdown.rstrip() + "\n"


def parse_log(
    filepath: str,
) -> tuple[
    dict[tuple[str, str, str], tuple[float | None, float | None]],
    dict[str, str],
]:
    """Parse a benchmark log and return (results, metadata).

    results: {(instance, strategy, mode): (median_ms or None, iqr_ms or None)}
    - (None, None) for SKIP'd instances
    metadata: thread environment variables found in the log
    """
    results: dict[tuple[str, str, str], tuple[float | None, float | None]] = {}
    current_mode: str | None = None
    current_strategy: str | None = None
    metadata: dict[str, str] = {}

    # Column layout flags (reset per file, but assumed constant throughout a file)
    has_iqr = False
    has_compile = False  # True when the log includes a "Compile (ms)" column

    engine: str | None = None
    rust_backend: str | None = None

    with open(filepath) as f:
        for line in f:
            line = line.rstrip()

            # --- Engine detection (header lines) ---
            if engine is None:
                if "tenferro-trace" in line.lower():
                    engine = "tenferro-trace"
                elif "tenferro-eager" in line.lower():
                    engine = "tenferro-eager"
                elif "tenferro-einsum" in line.lower():
                    engine = "tenferro-trace"
                elif "strided-opteinsum" in line.lower():
                    engine = "strided-opteinsum"
                elif "pytorch-cpu" in line.lower():
                    engine = "pytorch-cpu"
                elif "jax-cpu" in line.lower():
                    engine = "jax-cpu"

            # "Backend: tenferro-trace" / "Backend: strided-opteinsum(faer)"
            m = re.match(r"^Backend:\s+(.+)", line)
            if m and rust_backend is None:
                rust_backend = m.group(1).strip()

            # Thread-count metadata
            for var in (
                "OMP_NUM_THREADS",
                "RAYON_NUM_THREADS",
                "JULIA_NUM_THREADS",
            ):
                m2 = re.search(rf"{var}=(\d+)", line)
                if m2 and var not in metadata:
                    metadata[var] = m2.group(1)

            # --- Strategy header ---
            # Rust:  "Strategy: opt_flops"
            m = re.match(r"^Strategy:\s+(\w+)", line)
            if m:
                current_strategy = m.group(1)
                current_mode = rust_backend or engine or "unknown"
                continue

            # Julia: "Mode: omeinsum_path / Strategy: opt_flops"
            m = re.match(r"^Mode:\s+(\w+)\s*/\s*Strategy:\s+(\w+)", line)
            if m:
                current_mode = m.group(1)
                current_strategy = m.group(2)
                continue

            # --- Column header ---
            if line.startswith("Instance") and "IQR" in line:
                has_iqr = True
                # Detect whether a "Compile" column follows IQR.
                # Rust logs have it; Python logs do not.
                has_compile = "Compile" in line
                continue

            # Skip table decorators
            if line.startswith("Instance") or line.startswith("-"):
                continue

            # --- Data line ---
            # Expected columns (row-major order):
            #   name  tensors  log10flops  log2size  median  [iqr  [compile]]
            parts = line.split()
            if len(parts) >= 5 and current_mode and current_strategy:
                try:
                    name = parts[0]
                    last = parts[-1]

                    # SKIP detection: the status field ("SKIP") is always
                    # at index 4.  Using `last == "SKIP"` is unreliable when
                    # a Compile column is present, so we check parts[4] first.
                    if len(parts) > 4 and parts[4] == "SKIP":
                        key = (name, current_strategy, current_mode)
                        results[key] = (None, None)
                        continue

                    if has_iqr:
                        if has_compile and len(parts) >= 7:
                            # Rust format: name tensors log10f log2s median iqr compile
                            median_ms = float(parts[-3])
                            iqr_str = parts[-2]
                        else:
                            # Python format: name tensors log10f log2s median iqr
                            median_ms = float(parts[-2])
                            iqr_str = parts[-1]
                        iqr_ms: float | None = (
                            float(iqr_str) if iqr_str != "-" else None
                        )
                    else:
                        median_ms = float(last)
                        iqr_ms = None

                    key = (name, current_strategy, current_mode)
                    results[key] = (median_ms, iqr_ms)

                except (ValueError, IndexError):
                    continue

    return results, metadata


def format_markdown_table(
    all_results: dict[tuple[str, str, str], tuple[float | None, float | None]],
    metadata: dict[str, str],
) -> str:
    """Format results as a markdown table grouped by strategy."""
    instances = sorted(set(name for name, _, _ in all_results))
    strategies = sorted(set(strat for _, strat, _ in all_results))
    modes = sorted(set(mode for _, _, mode in all_results))

    has_iqr = any(
        iqr is not None for _, (m, iqr) in all_results.items() if m is not None
    )

    # Exclude modes that use a different contraction path (unfair comparison)
    excluded_modes = {"omeinsum_opt"}
    modes = [m for m in modes if m not in excluded_modes]

    # Column order: Rust engines first, then Python, then Julia
    preferred_order = [
        "tenferro-einsum",
        "tenferro-trace",
        "tenferro-eager",
        "strided-opteinsum(faer)",
        "strided-opteinsum(blas)",
        "strided-opteinsum",
        "pytorch-cpu",
        "jax-cpu",
        "omeinsum_path",
        "tensorops",
    ]
    core_modes = [
        "tenferro-eager",
        "tenferro-trace",
        "pytorch-cpu",
        "jax-cpu",
    ]
    for mode in core_modes:
        if all_results and mode not in modes:
            modes.append(mode)

    mode_order: list[str] = []
    for m in preferred_order:
        if m in modes:
            mode_order.append(m)
    for m in modes:
        if m not in mode_order:
            mode_order.append(m)

    mode_labels = {
        "tenferro-einsum": "tenferro-rs trace mode (ms)",
        "tenferro-trace": "tenferro-rs trace mode (ms)",
        "tenferro-eager": "tenferro-rs eager mode (ms)",
        "strided-opteinsum": "strided-rs (ms)",
        "strided-opteinsum(faer)": "strided-rs faer (ms)",
        "strided-opteinsum(blas)": "strided-rs OpenBLAS (ms)",
        "pytorch-cpu": "PyTorch Python (ms)",
        "jax-cpu": "JAX Python (XLA CPU dot) (ms)",
        "omeinsum_path": "OMEinsum.jl OpenBLAS (ms)",
        "omeinsum_opt": "OMEinsum.jl opt (ms)",
        "tensorops": "TensorOperations.jl (ms)",
    }

    lines: list[str] = []

    for strategy in strategies:
        lines.append(f"#### Strategy: {strategy}")
        lines.append("")
        thread_vars = ", ".join(
            f"{k}={metadata[k]}" for k in sorted(metadata) if k in metadata
        )
        timing_note = "Median ± IQR (ms)" if has_iqr else "Median time (ms)"
        if thread_vars:
            lines.append(f"{timing_note}. {thread_vars}.")
        else:
            lines.append(f"{timing_note}.")
        lines.append("")

        # Header row
        cols = [mode_labels.get(m, m) for m in mode_order]
        header = "| Instance | " + " | ".join(cols) + " |"
        separator = "|---|" + "|".join("---:" for _ in cols) + "|"
        lines.append(header)
        lines.append(separator)

        # Data rows
        for name in instances:
            # Find the minimum median value across modes for bold highlighting
            medians: dict[str, float] = {}
            for mode in mode_order:
                key = (name, strategy, mode)
                if key in all_results:
                    med = all_results[key][0]
                    if med is not None:
                        medians[mode] = med
            min_val = min(medians.values()) if medians else None

            row = [name]
            for mode in mode_order:
                key = (name, strategy, mode)
                if key in all_results:
                    median_ms, iqr_ms = all_results[key]
                    if median_ms is None:
                        formatted = "-"
                    elif has_iqr and iqr_ms is not None:
                        formatted = f"{median_ms:.3f} ± {iqr_ms:.3f}"
                    else:
                        formatted = f"{median_ms:.3f}"
                    if min_val is not None and median_ms == min_val:
                        formatted = f"**{formatted}**"
                    row.append(formatted)
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    return clean_markdown_eof("\n".join(lines))


def build_summary(
    all_results: dict[tuple[str, str, str], tuple[float | None, float | None]],
    metadata: dict[str, str],
    run_metadata: dict | None = None,
) -> dict:
    """Machine-readable sidecar for the same parsed log results."""
    header = report_summary.metadata_from_run_metadata(run_metadata)
    hardware = dict(header["hardware"])
    if not hardware.get("cpu"):
        cpu_model = (run_metadata or {}).get("environment", {}).get("cpu") if run_metadata else None
        hardware["cpu"] = cpu_model or "unknown"
    threads = sorted({int(value) for value in metadata.values() if str(value).isdigit()})
    if threads:
        hardware["threads"] = threads
    rows = []
    for (instance, strategy, mode), (median_ms, iqr_ms) in all_results.items():
        status = "ok" if median_ms is not None else "skipped"
        group = strategy if not threads else f"Threads: {threads[0]} / {strategy}"
        rows.append(
            report_summary.make_row(
                instance,
                mode,
                median_ms if status == "ok" else None,
                status=status,
                iqr=iqr_ms if status == "ok" else None,
                group=group,
                verified=status == "ok",
                reason="harness SKIP" if status == "skipped" else None,
            )
        )
    return report_summary.build_document(
        rows,
        target_profile=header["target_profile"],
        suite_id=header["suite_id"] or "cpu/einsum",
        timestamp=header["timestamp"],
        tenferro_commit=header["tenferro_commit"],
        hardware=hardware,
        unit="ms",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logfiles", nargs="+")
    parser.add_argument("--run-metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    all_results: dict[
        tuple[str, str, str], tuple[float | None, float | None]
    ] = {}
    all_metadata: dict[str, str] = {}
    for filepath in args.logfiles:
        results, metadata = parse_log(filepath)
        all_results.update(results)
        all_metadata.update(metadata)

    markdown = format_markdown_table(all_results, all_metadata)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
    else:
        sys.stdout.write(markdown)

    run_metadata = (
        report_summary.load_run_metadata(args.run_metadata)
        if args.run_metadata
        else report_summary.env_run_metadata()
    )
    report_summary.maybe_write(
        build_summary(all_results, all_metadata, run_metadata),
        summary_path=report_summary.summary_path_for_report(args.output) if args.output else None,
        env=os.environ,
    )


if __name__ == "__main__":
    main()
