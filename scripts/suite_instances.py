#!/usr/bin/env python3
"""Resolve benchmark instance IDs from a suite YAML file."""

from __future__ import annotations

import argparse
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

    if source_path.suffix in {".yaml", ".yml"}:
        base_ids = resolve_suite_instance_ids(source_path, project_dir=project_dir)
        allowed = set(include)
        ids = [instance_id for instance_id in base_ids if instance_id in allowed]
    else:
        ids = list(include)

    return [instance_id for instance_id in ids if instance_id not in exclude]


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
