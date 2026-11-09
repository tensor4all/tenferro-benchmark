#!/usr/bin/env python3
"""Resolve benchmark instance IDs from a suite YAML file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _load_suite(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: suite root must be a mapping")
    return data


def _resolve_problems(problems: Any, *, project_dir: Path) -> list[str]:
    if isinstance(problems, list):
        ids: list[str] = []
        for problem in problems:
            if not isinstance(problem, dict):
                raise ValueError("problems list entries must be mappings")
            problem_id = problem.get("id")
            if not isinstance(problem_id, str) or not problem_id:
                raise ValueError("each inline problem must have a non-empty id")
            ids.append(problem_id)
        return ids

    if not isinstance(problems, dict):
        raise ValueError("problems must be a list or selector mapping")

    source = problems.get("source")
    include = problems.get("include")
    exclude = set(problems.get("exclude") or [])

    if not isinstance(source, str) or not source:
        raise ValueError("problem selector requires a non-empty source")
    if not isinstance(include, list) or not include:
        raise ValueError("problem selector requires a non-empty include list")

    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = project_dir / source_path

    # `exists()` returns False for any OSError on Python 3.14, so a permission
    # error read as absence and the opt-in swallowed it. Only a real absence is
    # skippable -- the same rule the validator uses.
    try:
        source_path.stat()
    except FileNotFoundError:
        if problems.get("missing_source") != "skip":
            raise ValueError(f"{source_path}: include source did not load: no such file or directory")
        return [instance_id for instance_id in include if instance_id not in exclude]
    except OSError as error:
        raise ValueError(f"{source_path}: include source did not load: {error}") from error

    if source_path.is_dir() or source_path.suffix not in {".yaml", ".yml"}:
        # A JSON catalogue is read for its declared ids: copying `include`
        # unchecked was the same fail-open the validator rejected, and
        # run_all.sh collects through this resolver without validating.
        known = set(_catalogue_ids(source_path))
        missing = [instance_id for instance_id in include if instance_id not in known]
        if missing:
            raise ValueError(
                f"{source_path}: include id {missing[0]!r} is not in the source"
            )
        ids = list(include)
    elif source_path.suffix in {".yaml", ".yml"}:
        base_ids = resolve_suite_instance_ids(source_path, project_dir=project_dir)
        known = set(base_ids)
        missing = [instance_id for instance_id in include if instance_id not in known]
        if missing:
            # A selector must not collect fewer cases than it declares: the
            # validator rejects the same miss, but run_all.sh does not run it.
            raise ValueError(
                f"{source_path}: include id {missing[0]!r} is not in the source"
            )
        ids = [instance_id for instance_id in include if instance_id in known]
    else:
        ids = list(include)

    return [instance_id for instance_id in ids if instance_id not in exclude]



def _catalogue_ids(source_path: Path) -> list[str]:
    """Every record id a JSON catalogue file or directory declares."""
    ids: list[str] = []
    seen: set[str] = set()
    directory = source_path.is_dir()
    files = sorted(source_path.glob("*.json")) if directory else [source_path]
    for entry in files:
        data = json.loads(entry.read_text(encoding="utf-8"))
        if directory:
            # Mirrors `declared_cases`: each file is one record, named by `id`,
            # then `name`, then the file stem; a list root is several records.
            if isinstance(data, list):
                records, fallback = data, None
            else:
                records, fallback = [data], entry.stem
        else:
            # Mirrors `declared_cases`: a standalone catalogue is a list or a
            # {cases|patterns: [...]} mapping. A bare record declares no ids.
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = data.get("cases")
                if records is None:
                    records = data.get("patterns", [])
            else:
                records = []
            fallback = None
        if not isinstance(records, list):
            raise ValueError(f"{entry}: `cases` must be a list of records")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"{entry}: record {index} is not a mapping")
            record_id = record.get("id")
            if not record_id and fallback is not None:
                record_id = record.get("name") or fallback
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{entry}: record {index} has no non-empty id")
            if record_id in seen:
                raise ValueError(f"{entry}: duplicate problem id {record_id!r}")
            seen.add(record_id)
            ids.append(record_id)
    return ids


def resolve_suite_instance_ids(
    suite_file: Path,
    *,
    project_dir: Path = PROJECT_DIR,
) -> list[str]:
    suite_path = suite_file if suite_file.is_absolute() else project_dir / suite_file
    suite = _load_suite(suite_path)
    return _resolve_problems(suite["problems"], project_dir=project_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-file",
        type=Path,
        required=True,
        help="Suite YAML path relative to the repo root unless absolute",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "lines"),
        default="csv",
        help="Output one comma-separated line (csv) or one ID per line (lines)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids = resolve_suite_instance_ids(args.suite_file)
    if not ids:
        raise SystemExit(f"{args.suite_file}: resolved zero instance IDs")
    if args.format == "csv":
        print(",".join(ids))
    else:
        print("\n".join(ids))


if __name__ == "__main__":
    main()
