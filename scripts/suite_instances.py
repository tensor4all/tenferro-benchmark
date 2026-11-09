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


def _resolve_problems(
    problems: Any, *, project_dir: Path
) -> list[str]:
    if isinstance(problems, list):
        ids: list[str] = []
        for problem in problems:
            if not isinstance(problem, dict):
                raise ValueError("problems list entries must be mappings")
            problem_id = problem.get("id")
            if not isinstance(problem_id, str) or not problem_id:
                raise ValueError("each inline problem must have a non-empty id")
            if problem_id in ids:
                # The validator and the JSON catalogue path reject duplicates;
                # collecting them twice ran the same case twice.
                raise ValueError(f"duplicate problem id {problem_id!r}")
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
    repeated = [entry for index, entry in enumerate(include) if entry in include[:index]]
    if repeated:
        # The inline form rejects the same id twice; here it resolved to the case
        # twice, so run_all.sh collected a duplicate the validator accepted.
        raise ValueError(f"selector repeats problem id {repeated[0]!r}")

    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = project_dir / source_path

    # The same value the validator rejects, rejected here too: this used to be read
    # only inside the `FileNotFoundError` branch, so an unsupported policy passed
    # whenever the source happened to exist.
    missing_source = problems.get("missing_source")
    if missing_source is not None and missing_source != "skip":
        raise ValueError(
            f"{source_path}: missing_source {missing_source!r} is not supported (only skip)"
        )

    # `exists()` returns False for any OSError on Python 3.14, so a permission
    # error read as absence and the opt-in swallowed it. Only a real absence is
    # skippable -- the same rule the validator uses.
    try:
        source_path.stat()
    except FileNotFoundError:
        if missing_source != "skip":
            raise ValueError(f"{source_path}: include source did not load: no such file or directory")
        return [instance_id for instance_id in include if instance_id not in exclude]
    except OSError as error:
        raise ValueError(f"{source_path}: include source did not load: {error}") from error

    kind, document = _source_kind(source_path)
    if kind in {"catalogue", "directory"}:
        # A catalogue is read for its declared ids: copying `include` unchecked was
        # the same fail-open the validator rejected, and run_all.sh collects through
        # this resolver without validating.
        known = set(_catalogue_ids(source_path, document))
        missing = [instance_id for instance_id in include if instance_id not in known]
        if missing:
            raise ValueError(
                f"{source_path}: include id {missing[0]!r} is not in the source"
            )
        ids = list(include)
    elif kind == "suite":
        # A nested suite: main's behaviour, kept as it is. #426 owns making the
        # nested path as strict as the catalogue path (unknown ids, duplicates).
        base_ids = resolve_suite_instance_ids(source_path, project_dir=project_dir)
        allowed = set(include)
        ids = [instance_id for instance_id in base_ids if instance_id in allowed]
    else:
        raise ValueError(
            f"{source_path}: include source is neither a suite (a `problems` "
            "mapping) nor a catalogue (a `cases` list, a record list, or one record)"
        )

    return [instance_id for instance_id in ids if instance_id not in exclude]


def _source_kind(source_path: Path) -> tuple[str, Any]:
    """`("directory"|"catalogue"|"suite", parsed document)` for a selector source.

    Decided by structure, never by syntax or file name: JSON is a subset of YAML, so
    a successful `json.loads` says nothing about what the document means, and a
    suffix says nothing about its content. A `problems` mapping is a suite, a
    `cases`/`patterns` list is a catalogue, a bare `name` record is one instance.
    """
    if source_path.is_dir():
        return "directory", None
    text = _read_source(source_path)
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        # JSON is what every collector opens; YAML only adds the nested-suite form.
        document = yaml.safe_load(text)
    if isinstance(document, dict):
        if "problems" in document:
            return "suite", document
        if isinstance(document.get("cases"), list) or isinstance(document.get("patterns"), list):
            return "catalogue", document
        if isinstance(document.get("name"), str) and document["name"]:
            return "catalogue", document
    elif isinstance(document, list):
        return "catalogue", document
    return "unknown", document


def _read_source(source_path: Path) -> str:
    try:
        return source_path.read_text(encoding="utf-8")
    except (OSError, ValueError) as error:
        # `ValueError` covers `UnicodeDecodeError`; the validator reports it against
        # this file, so no ids can be collected from it either.
        raise ValueError(f"{source_path}: include source did not load: {error}") from error


def _catalogue_ids(source_path: Path, document: Any = None) -> list[str]:
    """Every record id a catalogue file or directory declares.

    A directory is read exactly as `benchmark_python.load_instances()` reads it --
    one record per file, keyed by `name`, arrays and nameless mappings skipped -- so
    the collector cannot collect an id the validator would not accept, or miss one
    it would. A standalone file is read by its own collector: a record list or a
    `{cases|patterns: [...]}` mapping keyed by `id`, or one `name` record.
    """
    ids: list[str] = []
    seen: set[str] = set()
    if source_path.is_dir():
        for entry in sorted(source_path.glob("*.json")):
            data = json.loads(entry.read_text(encoding="utf-8"))
            name = data.get("name") if isinstance(data, dict) else None
            if not isinstance(name, str) or not name:
                continue
            declared = data.get("id")
            if isinstance(declared, str) and declared and declared != name:
                raise ValueError(
                    f"{entry}: declares id {declared!r} and name {name!r}; "
                    "the loader collects by `name`"
                )
            if name in seen:
                raise ValueError(f"{entry}: duplicate problem id {name!r}")
            seen.add(name)
            ids.append(name)
        return ids

    data = document
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        if isinstance(data.get("name"), str) and data["name"] and not isinstance(
            data.get("cases"), list
        ):
            return [data["name"]]
        records = data.get("cases")
        if records is None:
            records = data.get("patterns", [])
    else:
        records = []
    if not isinstance(records, list):
        raise ValueError(f"{source_path}: `cases` must be a list of records")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{source_path}: record {index} is not a mapping")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{source_path}: record {index} has no non-empty id")
        if record_id in seen:
            raise ValueError(f"{source_path}: duplicate problem id {record_id!r}")
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
