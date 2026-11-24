#!/usr/bin/env python3
"""Publish the pages of run directories that belong to several suites.

``regenerate_report_summaries.py --publish`` renders the runs of one suite at a
time and refuses a mixed selection. A collection such as ``run_all.sh 1 4 8``
writes runs of many suites, so this groups them by suite (the suite each run
records, as ``render_run`` detects it) and publishes each group on its own,
exactly as ``publish_run.sh`` does per suite.

    uv run python scripts/publish_suite_runs.py RUN_DIR...
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import regenerate_report_summaries as regen  # noqa: E402


def group_by_suite(
    run_dirs: Iterable[Path], suite_of: Callable[[Path], str] = regen._suite_of
) -> dict[str, list[Path]]:
    """``{suite: [run dir, ...]}`` in input order; a run selected twice is an error."""
    groups: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    for candidate in run_dirs:
        run_dir = Path(candidate).resolve()
        if run_dir in seen:
            raise regen.PublishError(f"{run_dir}: selected twice")
        seen.add(run_dir)
        groups.setdefault(suite_of(run_dir), []).append(run_dir)
    return groups


def publish_by_suite(
    run_dirs: Iterable[Path],
    publish: Callable[[list[Path]], list[Path]] = regen.publish,
    suite_of: Callable[[Path], str] = regen._suite_of,
) -> list[Path]:
    written: list[Path] = []
    for suite, runs in sorted(group_by_suite(run_dirs, suite_of).items()):
        print(f"publish {suite}: " + " ".join(str(run) for run in runs))
        written += publish(runs)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        written = publish_by_suite(args.run_dirs)
    except regen.PublishError as error:
        print(f"publish: {error}", file=sys.stderr)
        return 1
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
