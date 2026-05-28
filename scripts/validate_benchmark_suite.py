#!/usr/bin/env python3
"""Validate GPU benchmark suite YAML files and result JSONL records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

PROJECT_DIR = Path(__file__).resolve().parents[1]
SUITE_SCHEMA = PROJECT_DIR / "schemas" / "benchmark-suite.schema.json"
RESULT_SCHEMA = PROJECT_DIR / "schemas" / "benchmark-result.schema.json"


class ValidationLoadError(Exception):
    """Raised when an input or schema file cannot be loaded."""


def load_json(path: Path) -> Any:
    try:
        with path.open() as fh:
            return json.load(fh)
    except OSError as exc:
        raise ValidationLoadError(f"{path}: failed to read JSON: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationLoadError(f"{path}: invalid JSON: {exc}") from exc


def load_yaml(path: Path) -> Any:
    try:
        with path.open() as fh:
            return yaml.safe_load(fh)
    except OSError as exc:
        raise ValidationLoadError(f"{path}: failed to read YAML: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValidationLoadError(f"{path}: invalid YAML: {exc}") from exc


def validator_for(schema_path: Path) -> Draft202012Validator:
    schema = load_json(schema_path)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValidationLoadError(f"{schema_path}: invalid JSON schema: {exc.message}") from exc
    return Draft202012Validator(schema)


def print_errors(path: Path, errors: list[Any]) -> None:
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path)
        prefix = f"{path}:{location}" if location else str(path)
        print(f"{prefix}: {error.message}", file=sys.stderr)


def validate_object(path: Path, obj: Any, validator: Draft202012Validator) -> bool:
    errors = sorted(validator.iter_errors(obj), key=lambda err: list(err.absolute_path))
    if errors:
        print_errors(path, errors)
        return False
    return True


def validate_suite(path: Path) -> bool:
    try:
        validator = validator_for(SUITE_SCHEMA)
        suite = load_yaml(path)
    except ValidationLoadError as exc:
        print(exc, file=sys.stderr)
        return False
    return validate_object(path, suite, validator)


def validate_results(path: Path) -> bool:
    try:
        validator = validator_for(RESULT_SCHEMA)
    except ValidationLoadError as exc:
        print(exc, file=sys.stderr)
        return False
    ok = True
    records = 0
    try:
        fh = path.open()
    except OSError as exc:
        print(f"{path}: failed to read JSONL: {exc}", file=sys.stderr)
        return False
    with fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"{path}:{line_no}: invalid JSON: {exc}", file=sys.stderr)
                ok = False
                continue
            records += 1
            if not validate_object(Path(f"{path}:{line_no}"), record, validator):
                ok = False
    if records == 0:
        print(f"{path}: no JSON records found", file=sys.stderr)
        ok = False
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--kind",
        choices=["suite", "result"],
        default="suite",
        help="Validate suite YAML or result JSONL records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ok = True
    for path in args.paths:
        if args.kind == "suite":
            ok = validate_suite(path) and ok
        else:
            ok = validate_results(path) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
