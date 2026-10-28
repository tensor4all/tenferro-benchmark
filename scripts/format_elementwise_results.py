#!/usr/bin/env python3
"""Format cpu/elementwise CSV results as Markdown.

Usage:
    python scripts/format_elementwise_results.py \\
        data/results/<target_profile>/cpu/elementwise/<timestamp>/elementwise_t1_<ts>.csv \\
        data/results/<target_profile>/cpu/elementwise/<timestamp>/elementwise_t4_<ts>.csv \\
        [--run-metadata data/results/<target_profile>/cpu/elementwise/<timestamp>/run.yaml]

Each input is the CSV written by src/bin/benchmark_cpu_elementwise.rs with the
columns suite,benchmark,dtype,shape,threads,pool_limit_bytes,median_ms,iqr_ms,gb_per_s.
Records are grouped by `threads` so a run at more than one thread count renders
one table per thread count. Within a thread count, every op is measured twice:
with the default CPU buffer-pool retention limit and with a raised one, so each
row carries its engine's actual `pool_limit_bytes`. The smaller limit of an op's
two rows is labelled `default` and the larger `raised`, and the `default/raised`
column on the `raised` row is the median ratio between them - that ratio is the
pool-policy slowdown the suite exists to expose, not a kernel speedup.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collect_cpu_info import collect_cpu_info, markdown as cpu_info_markdown  # noqa: E402

COLUMNS = [
    "suite",
    "benchmark",
    "dtype",
    "shape",
    "threads",
    "pool_limit_bytes",
    "median_ms",
    "iqr_ms",
    "gb_per_s",
]
OP_ORDER = ["sqrt", "rsqrt", "exp"]
DEFAULT_POOL_LIMIT_BYTES = 100 * 1024 * 1024


def clean_markdown_eof(markdown: str) -> str:
    return markdown.rstrip() + "\n"


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as fh:
        header = fh.readline().strip().split(",")
        if header != COLUMNS:
            raise SystemExit(
                f"{path}: unexpected CSV header {header!r}; expected {COLUMNS!r}"
            )
        for line_no, line in enumerate(fh, start=2):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split(",")
            if len(fields) != len(COLUMNS):
                raise SystemExit(f"{path}:{line_no}: expected {len(COLUMNS)} fields")
            records.append(
                {
                    "suite": fields[0],
                    "benchmark": fields[1],
                    "dtype": fields[2],
                    "shape": fields[3],
                    "threads": int(fields[4]),
                    "pool_limit_bytes": int(fields[5]),
                    "median_ms": float(fields[6]),
                    "iqr_ms": float(fields[7]),
                    "gb_per_s": float(fields[8]),
                }
            )
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


def format_bytes(count: int) -> str:
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if count and count % scale == 0:
            return f"{count // scale} {unit}"
    return f"{count} B"


def format_thread_section(threads: int, records: list[dict[str, Any]]) -> list[str]:
    by_op: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_op[record["benchmark"]].append(record)

    lines = [f"## Threads: {threads}", ""]
    lines.append("| op | pool | median ms | IQR ms | GB/s | default/raised |")
    lines.append("|---|---|---:|---:|---:|---:|")

    for op in sorted(by_op, key=lambda o: (OP_ORDER.index(o) if o in OP_ORDER else 99, o)):
        rows = sorted(by_op[op], key=lambda record: record["pool_limit_bytes"])
        # Arms are identified by their declared limit, never by row position: a
        # duplicate limit means the same thread count was measured twice.
        limits = [record["pool_limit_bytes"] for record in rows]
        if len(set(limits)) != len(limits):
            raise SystemExit(
                f"cpu/elementwise: duplicate pool limit for {op}: {limits}; "
                "run each thread count once"
            )
        default_median = (
            rows[0]["median_ms"]
            if len(rows) == 2 and limits[0] == DEFAULT_POOL_LIMIT_BYTES
            else None
        )
        for record in rows:
            label = (
                "default"
                if record["pool_limit_bytes"] == DEFAULT_POOL_LIMIT_BYTES
                else "raised"
            )
            ratio = "-"
            if label == "raised" and default_median is not None and record["median_ms"] > 0:
                ratio = f"{default_median / record['median_ms']:.2f}x"
            lines.append(
                f"| `{op}` | {label} ({format_bytes(record['pool_limit_bytes'])}) "
                f"| {record['median_ms']:.3f} | {record['iqr_ms']:.3f} "
                f"| {record['gb_per_s']:.2f} | {ratio} |"
            )

    lines.append("")
    return lines


def format_markdown(
    records: list[dict[str, Any]],
    run_metadata: dict[str, Any] | None = None,
) -> str:
    lines = ["# CPU Elementwise Benchmark Results", ""]
    if run_metadata:
        lines.extend(
            [
                f"- Target profile: `{run_metadata.get('target_profile', 'unknown')}`",
                f"- Suite: `{run_metadata.get('suite_id', 'cpu/elementwise')}`",
                f"- Suite file: `{run_metadata.get('suite_file', 'benchmarks/cpu/elementwise.yaml')}`",
                f"- Timestamp: `{run_metadata.get('timestamp', 'unknown')}`",
            ]
        )
        tenferro = run_metadata.get("tenferro_rs")
        if isinstance(tenferro, dict) and tenferro.get("commit"):
            lines.append(f"- tenferro-rs commit: `{tenferro['commit']}`")
        lines.append("")

    lines.append(cpu_info_markdown(collect_cpu_info()).rstrip())
    lines.append("")

    lines.append(
        "f64 `sqrt` / `rsqrt` / `exp` on 33_554_432 elements through the "
        "tenferro-rs eager CPU API. Each iteration drops the output tensor and "
        "only black-boxes its shape and dtype, so nothing retains it: under the "
        "default 100 MiB buffer-pool limit a 256 MiB output cannot come back "
        "through the pool, so the measured call pays for a new destination "
        "(allocation and, typically, first touch), while the raised 4 GiB limit "
        "lets the next call reuse the previous output buffer. There is "
        "deliberately no external reference backend: JAX and PyTorch differ in "
        "allocator and caching strategy, which is the difference this suite "
        "isolates. The huge-page mode above is part of the reading: with "
        "`madvise` a fresh large destination faults 4 KiB pages, with `always` the "
        "kernel may hand out 2 MiB pages and the default arm gets cheaper. See "
        "`docs/elementwise-suite.md` and tensor4all/tenferro-rs#2284."
    )
    lines.append("")
    lines.append(
        "Thread counts come from the engine: a one-thread run uses "
        "`CpuEngine::serial()`, a wider run gets its own pool, and every row "
        "records the width the engine reports. `RAYON_NUM_THREADS` / "
        "`OMP_NUM_THREADS` are also set so any library-level parallelism matches; "
        "no CPU-affinity pinning (`taskset` / `numactl`) is applied by the "
        "scripts, matching the other CPU suites. The controlled thread "
        "environment is recorded per thread count in the run's `run_t<N>.yaml`."
    )
    lines.append("")
    lines.append(
        "`default` is `DEFAULT_MAX_RETAINED_CAPACITY_BYTES` (100 MiB), set "
        "explicitly on the engine so an inherited "
        "`TENFERRO_BUFFER_POOL_MAX_RETAINED_BYTES` cannot move the baseline; "
        "`raised` is the same engine with `set_buffer_pool_limit_bytes(4 GiB)`. "
        "`IQR` is `p75 - p25`, `GB/s` is logical bytes (`2 * n * 8`) over the "
        "median, and `default/raised` is the **pool-policy slowdown**: the timed "
        "call covers the operation and the destination it has to take "
        "(allocation and first touch when the pool cannot help, a pool hand-back "
        "when it can), with recycling itself after the clock stops. The "
        "investigation in tensor4all/tenferro-rs#2284 attributes the dominant "
        "term to destination first touch."
    )
    lines.append("")

    by_threads: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_threads[record["threads"]].append(record)

    for threads in sorted(by_threads):
        lines.extend(format_thread_section(threads, by_threads[threads]))

    return clean_markdown_eof("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="CSV result files")
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
        print("No elementwise benchmark records found.", file=sys.stderr)
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
