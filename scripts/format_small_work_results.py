#!/usr/bin/env python3
"""Validate and render cpu/small_work JSONL records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from small_work import render_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = []
    for line_no, line in enumerate(args.records.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{args.records}:{line_no}: invalid JSON: {exc}") from exc
    try:
        report = render_report(records)
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"{args.records}: invalid small-work records: {exc}") from exc
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
