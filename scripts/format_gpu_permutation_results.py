#!/usr/bin/env python3
"""Format gpu/permutation JSON Lines results (Rust + Python runners) as Markdown.

Usage:
    python scripts/format_gpu_permutation_results.py \
        data/results/nvidia-gpu/gpu/permutation/<timestamp>/rust_output.jsonl \
        data/results/nvidia-gpu/gpu/permutation/<timestamp>/python_output.jsonl \
        [--run-metadata data/results/nvidia-gpu/gpu/permutation/<timestamp>/run.yaml]

Each input is a JSON Lines file where every line is one record produced by
src/bin/benchmark_gpu_permutation.rs or scripts/benchmark_gpu_permutation_python.py
(see docs/gpu-permutation-suite.md for the record shape). Modeled closely on
scripts/format_permutation_results.py (the cpu/permutation formatter), but:

- GPU info is embedded via scripts/collect_gpu_info.py instead of CPU info
  (mirrors how scripts/format_gpu_results.py embeds GPU info).
- There is no thread-count dimension on this suite: records carry `device`
  (a GPU name string) instead of `threads`, so this renders a single table
  rather than one table per thread count.
- Column order: tenferro-cuda-transpose, tenferro-cuda-to-contiguous,
  cutensor, pytorch-cuda, jax-cuda, memcpy-d2d.

Missing backends are shown as `-`; the fastest backend per row is bolded.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collect_gpu_info import markdown as gpu_info_markdown, resolve_gpu_info  # noqa: E402

BACKEND_ORDER = [
    "tenferro-cuda-transpose",
    "tenferro-cuda-to-contiguous",
    "cutensor",
    "pytorch-cuda",
    "jax-cuda",
    "memcpy-d2d",
]

BACKEND_LABELS = {
    "tenferro-cuda-transpose": "tenferro-rs CUDA transpose (ms)",
    "tenferro-cuda-to-contiguous": "tenferro-rs CUDA to_contiguous (ms)",
    "cutensor": "cuTENSOR (ms)",
    "pytorch-cuda": "PyTorch CUDA (ms)",
    "jax-cuda": "JAX CUDA (ms)",
    "memcpy-d2d": "memcpy D2D (ms)",
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


def format_table(records: list[dict[str, Any]]) -> list[str]:
    by_pattern: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (record["pattern_id"], record.get("label", ""), record.get("dtype", "f64"))
        by_pattern[key][record["backend"]] = record

    lines = [
        "Median (p25 / p75) in ms. Missing backends are shown as `-`; the "
        "fastest backend per pattern is **bolded**.",
        "",
        "| pattern | label | " + " | ".join(BACKEND_LABELS[b] for b in BACKEND_ORDER) + " |",
        "|---|---|" + "|".join("---:" for _ in BACKEND_ORDER) + "|",
    ]

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
    lines = ["# GPU Permutation Benchmark Results", ""]
    if run_metadata:
        lines.extend(
            [
                f"- Target profile: `{run_metadata.get('target_profile', 'unknown')}`",
                f"- Suite: `{run_metadata.get('suite_id', 'gpu/permutation')}`",
                f"- Suite file: `{run_metadata.get('suite_file', 'benchmarks/gpu/permutation.yaml')}`",
                f"- Timestamp: `{run_metadata.get('timestamp', 'unknown')}`",
            ]
        )
        tenferro = run_metadata.get("tenferro_rs")
        if isinstance(tenferro, dict) and tenferro.get("commit"):
            lines.append(f"- tenferro-rs commit: `{tenferro['commit']}`")
        lines.append("")

    gpu_info = resolve_gpu_info(records, run_metadata)
    gpu_block = gpu_info_markdown(gpu_info).rstrip()
    if gpu_block:
        lines.append(gpu_block)
        lines.append("")

    lines.append(
        "`tenferro-cuda-transpose` is the eager `TensorStructural::transpose` op on "
        "`CudaBackend` (compact col-major input only); `tenferro-cuda-to-contiguous` is "
        "`TypedTensor::backend_region_view` (source layout) + "
        "`TypedTensorView::transpose_view(perm)` + "
        "`TensorViewCanonicalization::to_contiguous` "
        "(accepts arbitrary source strides). Both allocate a fresh device tensor on every "
        "call; `cutensor`, `pytorch-cuda`, and `memcpy-d2d` reuse a destination buffer "
        "allocated once per pattern. `jax-cuda` allocates a fresh output array per call "
        "(functional `jnp.transpose`), cannot express arbitrary source strides (so it "
        "does not participate in the explicit-stride pattern), and materializes its "
        "output in XLA's default row-major layout -- the public JAX API cannot request a "
        "col-major output, so at the byte level `jax-cuda` performs the "
        "reversal-conjugate permutation task (same shape multiset and mirrored stride "
        "structure, equivalent difficulty class) rather than the identical col-major "
        "write the other columns perform (see docs/gpu-permutation-suite.md). "
        "`jax-cuda` is additionally excluded from the rank-24 contiguous pattern "
        "(`tn_light_415_24d_contiguous_same_perm`) because XLA's jit compilation of the "
        "rank-24 transpose does not complete in practical time (>40 min observed). "
        "`memcpy-d2d` only "
        "participates in the contiguous identity-permutation baseline pattern. "
        "Correctness is verified against a host-computed naive reference, downloaded "
        "outside any timed region, before any timing; a `FAILED` cell means that "
        "backend's output did not match the reference for that pattern. A `skipped` cell "
        "means the backend does not participate in that pattern's semantics (or, for "
        "`cutensor`, that the installed cuTENSOR library rejected the pattern's rank at "
        "runtime) -- both are reported as `skipped` rather than a failure."
    )
    lines.append("")

    lines.extend(format_table(records))

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
    for input_path in args.inputs:
        if input_path.exists():
            records.extend(load_records(input_path))
        else:
            print(f"WARNING: {input_path} does not exist; skipping", file=sys.stderr)

    if not records:
        print("No gpu/permutation benchmark records found.", file=sys.stderr)
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
