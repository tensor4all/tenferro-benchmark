#!/usr/bin/env python3
"""Record where a delegated GPU suite's wall time went.

The generic driver (`scripts/run_gpu_suite.sh`) times each backend process and
appends it to the run's `wall_backends.jsonl`. A delegated collector owns its own
loop, so before this it published an empty file and its suites could only be
described as "collection time". This appends the same per-backend row from inside
the delegate, and keeps the non-backend phases (fixture preparation, a Rust build,
the separate CPU verification pass, report generation) in `wall_phases.json`,
where they cannot be mistaken for an arm's process time.

Both files are wall-clock observations. Nothing here changes a measurement, a
status or an exit code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARM_SCOPE = ("backend process wall time includes startup, preparation, validation "
             "and samples; recorded by the suite's own collector")


def _seconds(start_ns: int, end_ns: int) -> float:
    if end_ns < start_ns:
        raise ValueError("wall interval ends before it starts")
    return (end_ns - start_ns) / 1e9


def arm(output: Path, backend: str, start_ns: int, end_ns: int, build_seconds: float) -> dict:
    """Append one backend row in the shape the generic driver writes."""
    if build_seconds < 0:
        raise ValueError("build_seconds must not be negative")
    row = {"backend": backend, "build_seconds": build_seconds,
           "process_seconds": _seconds(start_ns, end_ns), "scope": ARM_SCOPE}
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    return row


def phase(output: Path, name: str, start_ns: int, end_ns: int) -> dict:
    """Add one non-backend phase to the run's phase file, summing repeats."""
    recorded = {}
    if output.exists():
        recorded = json.loads(output.read_text() or "{}")
    phases = recorded.setdefault("phases_seconds", {})
    phases[name] = round(phases.get(name, 0.0) + _seconds(start_ns, end_ns), 3)
    recorded["scope"] = ("wall clock of the stages a suite runs outside its backend "
                         "processes: fixture preparation, builds, separate verification "
                         "passes and report generation")
    output.write_text(json.dumps(recorded, indent=2) + "\n")
    return phases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["arm", "phase"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend")
    parser.add_argument("--phase")
    parser.add_argument("--start-ns", type=int, required=True)
    parser.add_argument("--end-ns", type=int, required=True)
    parser.add_argument("--build-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.kind == "arm":
        if not args.backend:
            parser.error("arm needs --backend")
        arm(args.output, args.backend, args.start_ns, args.end_ns, args.build_seconds)
    else:
        if not args.phase:
            parser.error("phase needs --phase")
        phase(args.output, args.phase, args.start_ns, args.end_ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
