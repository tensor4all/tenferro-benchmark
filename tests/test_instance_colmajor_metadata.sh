#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

uv run python - <<'PY'
import json
from pathlib import Path


def rowmajor_to_colmajor_format(format_string: str) -> str:
    inputs_str, output_str = format_string.split("->")
    inputs = inputs_str.split(",")
    return ",".join(operand[::-1] for operand in inputs) + "->" + output_str[::-1]


errors: list[str] = []
for path in sorted(Path("data/instances").glob("*.json")):
    with path.open() as fh:
        instance = json.load(fh)
    rowmajor_format = instance.get("format_string_rowmajor") or instance["format_string"]
    expected_format = rowmajor_to_colmajor_format(rowmajor_format)
    expected_shapes = [list(reversed(shape)) for shape in instance["shapes"]]
    if instance.get("format_string_colmajor") != expected_format:
        errors.append(
            f"{path}: format_string_colmajor={instance.get('format_string_colmajor')!r}, "
            f"expected {expected_format!r}"
        )
    if instance.get("shapes_colmajor") != expected_shapes:
        errors.append(
            f"{path}: shapes_colmajor={instance.get('shapes_colmajor')!r}, "
            f"expected {expected_shapes!r}"
        )

if errors:
    raise SystemExit("\n".join(errors))
PY
