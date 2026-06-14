#!/usr/bin/env python3
"""Import einsum benchmark instance metadata from extern/tenferro-einsum-benchmark.

Copies JSON metadata from the upstream benchmark checkout into data/instances/,
normalizing column-major fields and preserving repo-local intent/notes metadata.

Repo-only diagnostic instances (for example nary_matmul_chain_64) are left
untouched when they are absent from the upstream export.

Usage:
    uv run python scripts/import_einsum_dataset.py
    uv run python scripts/import_einsum_dataset.py --source extern/tenferro-einsum-benchmark/data/instances
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE = PROJECT_DIR / "extern" / "tenferro-einsum-benchmark" / "data" / "instances"
DEFAULT_DEST = PROJECT_DIR / "data" / "instances"


def convert_format_string_to_colmajor(format_string: str) -> str:
    inputs_str, output_str = format_string.split("->")
    inputs = inputs_str.split(",")
    reversed_inputs = [operand[::-1] for operand in inputs]
    reversed_output = output_str[::-1]
    return ",".join(reversed_inputs) + "->" + reversed_output


def normalize_instance(raw: dict, existing: dict | None) -> dict:
    rowmajor = raw.get("format_string_rowmajor") or raw["format_string"]
    meta = dict(raw)
    meta["format_string"] = rowmajor
    meta["format_string_rowmajor"] = rowmajor
    meta["format_string_colmajor"] = convert_format_string_to_colmajor(rowmajor)
    meta["shapes_colmajor"] = [list(reversed(shape)) for shape in meta["shapes"]]

    if existing:
        for key in ("intent", "notes"):
            if key in existing and existing[key]:
                meta[key] = existing[key]

    return meta


def import_instances(source_dir: Path, dest_dir: Path) -> list[str]:
    if not source_dir.is_dir():
        raise SystemExit(
            f"Source directory not found: {source_dir}\n"
            "Clone https://github.com/tensor4all/tenferro-einsum-benchmark "
            "into extern/tenferro-einsum-benchmark first."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for source_path in sorted(source_dir.glob("*.json")):
        with source_path.open() as fh:
            raw = json.load(fh)

        dest_path = dest_dir / source_path.name
        existing = None
        if dest_path.exists():
            with dest_path.open() as fh:
                existing = json.load(fh)

        meta = normalize_instance(raw, existing)
        with dest_path.open("w") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        written.append(source_path.name)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Upstream instance directory (default: {DEFAULT_SOURCE.relative_to(PROJECT_DIR)})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST.relative_to(PROJECT_DIR)})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = import_instances(args.source.resolve(), args.dest.resolve())
    print(f"Imported {len(written)} instances into {args.dest}:")
    for name in written:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
