#!/usr/bin/env python3
"""Finalize a run directory: merged case status and the command wall.

The finalize step writes collection facts only, so it also runs when reporting
is disabled (``BENCHMARK_RAW_ONLY=1``): the per-thread ``case_status_t<N>.json``
files are merged into ``case_status.json`` and ``wall_command.json`` records how
long the command took through collection.  Nothing here renders a report, a
summary or a published page.

Usage::

    uv run python scripts/finalize_run.py --run-dir data/results/<profile>/cpu/<suite>/<ts>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_wall  # noqa: E402
import case_status  # noqa: E402


def merge_thread_case_status(run_dir: Path) -> dict | None:
    """Merge the per-thread status files a runner wrote for this run directory."""
    run_dir = Path(run_dir)
    parts = [
        json.loads(path.read_text())
        for path in sorted(run_dir.glob("case_status_t*.json"))
    ]
    return case_status.merge_case_status(parts) if parts else None


def finalize(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    written: list[Path] = []
    status = merge_thread_case_status(run_dir)
    if status is not None:
        path = run_dir / "case_status.json"
        case_status.write_case_status(path, status)
        written.append(path)
    benchmark_wall.record_collection_wall(run_dir)
    wall = run_dir / "wall_command.json"
    if wall.is_file():
        written.append(wall)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        written = finalize(args.run_dir)
    except FileNotFoundError as error:
        print(f"finalize_run: {error}: not a run directory", file=sys.stderr)
        return 1
    for path in written:
        print(f"finalized {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
