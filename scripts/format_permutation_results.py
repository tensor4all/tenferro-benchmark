#!/usr/bin/env python3
"""Format cpu/permutation JSON Lines results (Rust + Julia runners) as Markdown.

Usage:
    python scripts/format_permutation_results.py \
        data/results/<target_profile>/cpu/permutation/<timestamp>/rust_output.jsonl \
        data/results/<target_profile>/cpu/permutation/<timestamp>/julia_output.jsonl \
        [--run-metadata data/results/<target_profile>/cpu/permutation/<timestamp>/run.yaml]

Each input is a JSON Lines file where every line is one record produced by
src/bin/benchmark_permutation.rs or scripts/benchmark_permutation.jl, matching
schemas/permutation-result.schema.json (see docs/permutation-suite.md for the
record shape). Every input file is validated against that schema before
formatting; an invalid record aborts the whole run (records are
machine-generated, so a schema violation means the runners and the schema
have drifted apart, not a one-off data issue to warn past). Records are
grouped by `threads` so a suite run at more than one thread count renders one
table per thread count. Missing backends are shown as `-`; the fastest
backend per row is bolded.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collect_cpu_info import collect_cpu_info, markdown as cpu_info_markdown  # noqa: E402
from validate_benchmark_suite import (  # noqa: E402
    PERMUTATION_RESULT_SCHEMA,
    validate_results as validate_permutation_result_jsonl,
)

BACKEND_ORDER = [
    "naive",
    "tenferro-transpose",
    "tenferro-to-contiguous",
    "hptt",
    "strided-rs",
    "julia-base",
    "strided-jl",
    "memcpy",
]

BACKEND_LABELS = {
    "naive": "naive odometer (ms)",
    "tenferro-transpose": "tenferro-rs transpose (ms)",
    "tenferro-to-contiguous": "tenferro-rs to_contiguous (ms)",
    "hptt": "HPTT (ms)",
    "strided-rs": "strided-rs (ms)",
    "julia-base": "Julia Base (ms)",
    "strided-jl": "Strided.jl (ms)",
    "memcpy": "memcpy (ms)",
}


def clean_markdown_eof(markdown: str) -> str:
    return markdown.rstrip() + "\n"


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                print(f"{path}:{line_no}: skipping invalid JSON: {exc}", file=sys.stderr)
    return records


def load_run_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        import yaml
    except ImportError:
        print(f"WARNING: PyYAML unavailable; skipping {path}", file=sys.stderr)
        return None
    with path.open() as fh:
        metadata = yaml.safe_load(fh)
    return metadata if isinstance(metadata, dict) else None


def format_cell(record: dict[str, Any] | None) -> tuple[str, float | None]:
    if record is None:
        return "-", None
    status = record.get("status")
    if status == "ok":
        median = record.get("median_ms")
        if median is None:
            return "ok (missing median)", None
        p25 = record.get("p25_ms")
        p75 = record.get("p75_ms")
        if p25 is not None and p75 is not None:
            text = f"{median:.3f} ({p25:.3f} / {p75:.3f})"
        else:
            text = f"{median:.3f}"
        return text, float(median)
    if status == "verification_failed":
        return "FAILED", None
    if status == "skipped":
        note = record.get("notes")
        return f"skipped ({note})" if note else "skipped", None
    return str(status or "-"), None


def bold_fastest(cells: dict[str, tuple[str, float | None]]) -> dict[str, str]:
    fastest_backend = None
    fastest_value = None
    for backend, (_, value) in cells.items():
        if value is None:
            continue
        if fastest_value is None or value < fastest_value:
            fastest_value = value
            fastest_backend = backend

    out: dict[str, str] = {}
    for backend, (text, _) in cells.items():
        if backend == fastest_backend:
            out[backend] = f"**{text}**"
        else:
            out[backend] = text
    return out


def format_thread_section(threads: str, records: list[dict[str, Any]]) -> list[str]:
    by_pattern: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (record["pattern_id"], record.get("label", ""), record.get("dtype", "f64"))
        by_pattern[key][record["backend"]] = record

    lines = [f"## Threads: {threads}", ""]
    lines.append(
        "Median (p25 / p75) in ms. Missing backends are shown as `-`; the "
        "fastest backend per pattern is **bolded**."
    )
    lines.append("")
    lines.append(
        "| pattern | label | " + " | ".join(BACKEND_LABELS[b] for b in BACKEND_ORDER) + " |"
    )
    lines.append("|---|---|" + "|".join("---:" for _ in BACKEND_ORDER) + "|")

    for key in sorted(by_pattern):
        pattern_id, label, _dtype = key
        backends = by_pattern[key]
        cells = {b: format_cell(backends.get(b)) for b in BACKEND_ORDER}
        rendered = bold_fastest(cells)
        row = [f"`{pattern_id}`", label] + [rendered[b] for b in BACKEND_ORDER]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return lines


def format_markdown(
    records: list[dict[str, Any]],
    run_metadata: dict[str, Any] | None = None,
) -> str:
    lines = ["# CPU Permutation Benchmark Results", ""]
    if run_metadata:
        lines.extend(
            [
                f"- Target profile: `{run_metadata.get('target_profile', 'unknown')}`",
                f"- Suite: `{run_metadata.get('suite_id', 'cpu/permutation')}`",
                f"- Suite file: `{run_metadata.get('suite_file', 'benchmarks/cpu/permutation.yaml')}`",
                f"- Timestamp: `{run_metadata.get('timestamp', 'unknown')}`",
            ]
        )
        tenferro = run_metadata.get("tenferro_rs")
        if isinstance(tenferro, dict) and tenferro.get("commit"):
            lines.append(f"- tenferro-rs commit: `{tenferro['commit']}`")
        lines.append("")

    lines.append(cpu_info_markdown(collect_cpu_info()).rstrip())
    lines.append("")

    if platform.system() == "Darwin":
        lines.append(
            "CPU pinning is unavailable on macOS; thread counts are controlled only "
            "via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`."
        )
        lines.append("")
    elif platform.system() == "Linux":
        lines.append(
            "On the Linux CPU devcontainer, thread counts are controlled via "
            "`RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no "
            "CPU-affinity pinning (`taskset` / `numactl`) is applied, matching the "
            "repository devcontainer convention. The controlled thread environment "
            "is recorded per thread count in the run's `run_t<N>.yaml`."
        )
        lines.append("")

    lines.append(
        "`tenferro-transpose` is the eager `CpuBackend::transpose` structural op "
        "(compact col-major input only); `tenferro-to-contiguous` is "
        "`TypedTensorView::transpose_view` + `CpuBackend::to_contiguous` (accepts "
        "arbitrary source strides). Every backend allocates a fresh destination "
        "inside each timed call, so the table compares allocation-inclusive "
        "end-to-end materialization rather than destination-reuse copy kernels. "
        "`hptt` only participates in patterns with a contiguous source and "
        "destination. Correctness is verified against the `naive` odometer "
        "reference before any timing; a `FAILED` cell means that backend's output "
        "did not match the reference for that pattern."
    )
    lines.append("")

    by_threads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        threads = str(record.get("threads", "unknown"))
        by_threads[threads].append(record)

    for threads in sorted(by_threads, key=lambda t: (len(t), t)):
        lines.extend(format_thread_section(threads, by_threads[threads]))

    return clean_markdown_eof("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="JSON Lines result files")
    parser.add_argument("--run-metadata", type=Path, default=None, help="run.yaml path")
    parser.add_argument("--output", type=Path, default=None, help="output path (default stdout)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    schema_ok = True
    for input_path in args.inputs:
        if input_path.exists():
            if not validate_permutation_result_jsonl(input_path, PERMUTATION_RESULT_SCHEMA):
                schema_ok = False
            records.extend(load_records(input_path))
        else:
            print(f"WARNING: {input_path} does not exist; skipping", file=sys.stderr)

    if not schema_ok:
        print(
            "One or more input files do not match "
            "schemas/permutation-result.schema.json; aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not records:
        print("No permutation benchmark records found.", file=sys.stderr)
        sys.exit(1)

    run_metadata = load_run_metadata(args.run_metadata)
    markdown = format_markdown(records, run_metadata)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
    else:
        sys.stdout.write(markdown)


if __name__ == "__main__":
    main()
