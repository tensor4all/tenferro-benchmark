#!/usr/bin/env python3
"""Resolve benchmark instance IDs from a suite YAML file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]

# The six shapes a selector source can have.  `resolve_source()` is the only place
# that decides which one it is, so the validator, this resolver and the collector
# cannot read the same file as different things.
INSTANCE_RECORD = "instance_record"
CATALOGUE_CASES = "catalogue_cases"
CATALOGUE_PATTERNS = "catalogue_patterns"
SUITE_DOCUMENT = "suite_document"
DIRECTORY = "directory"
BROKEN = "broken"
SOURCE_KINDS = frozenset(
    {
        INSTANCE_RECORD,
        CATALOGUE_CASES,
        CATALOGUE_PATTERNS,
        SUITE_DOCUMENT,
        DIRECTORY,
        BROKEN,
    }
)


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
        # twice, so run_all.sh collected a duplicate the validator accepted.  Same
        # wording as the validator's, so the two paths read as one rule.
        raise ValueError(
            f"{source}: include repeats id {repeated[0]!r}; a selector declares each id once"
        )

    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = project_dir / source_path

    # Before the absence branch: a skipped source used to return the entries
    # unchecked, so `include: [a, 7]` resolved while the validator rejected it.
    for entry in include:
        if not isinstance(entry, str) or not entry:
            raise ValueError(include_missing_error(source_path, entry))

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

    # The classification and the record-level checks live in `resolve_source()`,
    # which the validator calls too: this resolver runs without validating, so a
    # source it reads differently from the validator collects ids nobody declared.
    unreadable: dict[str, str] = {}
    kind, records, error = resolve_source(source_path, unreadable)
    if error is not None:
        raise ValueError(error)
    if kind == SUITE_DOCUMENT:
        # A nested suite: main's behaviour, kept as it is. #426 owns making the
        # nested path as strict as the catalogue path (duplicates, exclusions) --
        # but an id it does not declare is the same fail-open as in a catalogue, and
        # the validator rejects it, so this path does too.
        try:
            base_ids = resolve_suite_instance_ids(source_path, project_dir=project_dir)
        except RecursionError as error:
            raise ValueError(
                f"{source_path}: include source did not load: {error}"
            ) from error
        known = set(base_ids)
    else:
        # A catalogue is read for its declared ids: copying `include` unchecked was
        # the same fail-open the validator rejected, and run_all.sh collects through
        # this resolver without validating.
        known = {record["id"] for record in records}

    missing = [instance_id for instance_id in include if instance_id not in known]
    if missing:
        raise ValueError(include_missing_error(source_path, missing[0], unreadable))
    return [instance_id for instance_id in include if instance_id not in exclude]


def include_missing_error(
    source_path: Path, entry: str, unreadable: dict[str, str] | None = None
) -> str:
    """The message for an `include` id its source does not declare.

    Shared with the validator so the two paths report the same rule in the same
    words, including which file holds an id that is not a record to the loader.
    """
    member = (unreadable or {}).get(entry)
    if member is not None:
        return (
            f"{source_path}: include id {entry!r} is declared in {member}, which is "
            "not one record: the collector reads one record per file, keyed by `name`"
        )
    return f"{source_path}: include id {entry!r} is not in the source"


def resolve_source(
    source_path: Path, unreadable: dict[str, str] | None = None
) -> tuple[str, list[dict[str, Any]], str | None]:
    """`(kind, records, error)` -- the one place that decides what a source is.

    The validator, this resolver and `benchmark_python.load_instances()` each used
    to read the same file their own way, so one source could be a broken JSON
    catalogue to the validator, a YAML list of cases to the resolver and one
    instance to the collector.  The kind is decided by the document's *structure*:
    a `problems` mapping is a suite, a `cases`/`patterns` list (or a bare array) is a
    catalogue, one `name` mapping is an instance record.

    A document that starts like JSON (`{` or `[`) is read as JSON only.  YAML is not
    a fallback for a JSON parse error: PyYAML accepts the trailing comma that makes
    `{"cases": [{"id": "a",}],}` invalid JSON, so falling back turned a broken
    catalogue into a valid-looking one.  A `.yaml`/`.yml` file that does not start
    like JSON is read as YAML, which is where a suite document lives.  The
    consequence is deliberate: a YAML *flow* document (`{id: a}`) has to be written
    with JSON's quoted keys to be read as a source, because a `{` that does not
    parse as JSON is a broken JSON document and not a YAML one.

    `records` are the rows the source declares, each a mapping with a non-empty
    string `id`; `error` is `None` unless the kind is `BROKEN`, where it names the
    file and the reason.  `SUITE_DOCUMENT` returns `[document]`: a suite is not a
    catalogue, so the caller resolves its own selector against it.  For `DIRECTORY`
    the ids advertised by members that are not one record (a JSON array, a nameless
    mapping) are reported into `unreadable` (id -> file name) so the caller can name
    the file when a selector asks for one.
    """
    try:
        is_directory = source_path.is_dir()
    except (OSError, ValueError) as error:
        return BROKEN, [], f"{source_path}: include source did not load: {error}"
    if is_directory:
        return _resolve_directory(source_path, unreadable)
    try:
        text = source_path.read_text(encoding="utf-8")
    except (OSError, ValueError) as error:
        # `ValueError` covers `UnicodeDecodeError`; the validator reports it against
        # this file, so no ids can be collected from it either.
        return BROKEN, [], f"{source_path}: include source did not load: {error}"
    try:
        document = _parse_document(text, source_path)
    except ValueError as error:
        return BROKEN, [], str(error)
    except RecursionError as error:
        # A pathological nesting depth in the JSON/YAML parser; it must not
        # escape the per-source boundary and stop the remaining inputs.
        return BROKEN, [], f"{source_path}: include source did not load: {error}"
    return _classify(document, source_path)


def _parse_document(text: str, source_path: Path) -> Any:
    """The parsed document, or `ValueError` naming the file.

    Not `json.loads(...) except JSONDecodeError: yaml.safe_load(...)`: that fallback
    is the bug this function exists to remove.
    """
    if text.lstrip("\ufeff \t\r\n").startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{source_path}: include source did not load: {error}"
            ) from error
    if source_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ValueError(
                f"{source_path}: include source did not load: {error}"
            ) from error
        except Exception as error:  # noqa: BLE001 -- one file must not stop the rest
            # PyYAML's implicit resolvers can fail outside `YAMLError`: 6.0.3's
            # `!!timestamp` resolver raises `AttributeError: 'NoneType' object has
            # no attribute `groupdict` for an unparsable value, so a handler keyed
            # only on `YAMLError` lets one malformed source abort the validator's
            # remaining inputs. Report it against this file instead.
            raise ValueError(
                f"{source_path}: include source did not load: "
                f"{type(error).__name__}: {error}"
            ) from error
    raise ValueError(
        f"{source_path}: include source is neither a JSON document nor a "
        ".yaml/.yml one"
    )


def _classify(
    document: Any, source_path: Path
) -> tuple[str, list[dict[str, Any]], str | None]:
    """`(kind, records, error)` for one parsed document."""
    if isinstance(document, dict):
        cases = document.get("cases")
        patterns = document.get("patterns")
        shapes = [
            name
            for name, held in (
                ("problems", "problems" in document),
                ("cases", isinstance(cases, list)),
                ("patterns", isinstance(patterns, list)),
            )
            if held
        ]
        if len(shapes) > 1:
            # A document that is a suite *and* a catalogue has two readings; the
            # kind is not guessed, the file says which one it is.
            return BROKEN, [], (
                f"{source_path}: declares {' and '.join(f'`{name}`' for name in shapes)}; "
                "a source is one of a suite and a catalogue, so keep one shape"
            )
        if shapes == ["problems"]:
            return SUITE_DOCUMENT, [document], None
        if shapes == ["patterns"]:
            return _keyed(CATALOGUE_PATTERNS, patterns, source_path)
        if shapes == ["cases"]:
            return _keyed(CATALOGUE_CASES, cases, source_path)
        name = document.get("name")
        if isinstance(name, str) and name:
            # One instance record, keyed by `name` -- which is how
            # `benchmark_python.load_instances()` collects it.
            declared = document.get("id")
            if isinstance(declared, str) and declared and declared != name:
                return BROKEN, [], (
                    f"{source_path}: declares id {declared!r} and name {name!r}; "
                    "the loader collects by `name`"
                )
            return INSTANCE_RECORD, [{**document, "id": name}], None
        return BROKEN, [], (
            f"{source_path}: include source did not load: the root is neither a "
            "suite (a `problems` mapping) nor a catalogue (a `cases`/`patterns` "
            "list, a record list, or one `name` record)"
        )
    if isinstance(document, list):
        # A bare array is a `cases` catalogue without its wrapper: the committed
        # `data/instances/{perf_issues,small_work}.json` are exactly this.
        return _keyed(CATALOGUE_CASES, document, source_path)
    return BROKEN, [], f"{source_path}: include source did not load: the root is not a mapping"


def _keyed(
    kind: str, records: list[Any], source_path: Path
) -> tuple[str, list[dict[str, Any]], str | None]:
    """`(kind, records, None)` once every record has a unique, non-empty id."""
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            return BROKEN, [], (
                f"{source_path}: include source did not load: record {index} is not a mapping"
            )
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            return BROKEN, [], (
                f"{source_path}: include source did not load: a case without a "
                f"non-empty id (record {index})"
            )
        if record_id in seen:
            return BROKEN, [], (
                f"{source_path}: include source did not load: declares duplicate "
                f"problem id {record_id!r}"
            )
        seen.add(record_id)
    return kind, records, None


def _resolve_directory(
    directory: Path, unreadable: dict[str, str] | None
) -> tuple[str, list[dict[str, Any]], str | None]:
    """One record per `*.json` member, read as the name-keyed loader reads it.

    `benchmark_python.load_instances()` opens every `*.json` member, keeps the ones
    that are a mapping with a non-empty `name`, and warns the rest away.  Doing it
    here means a directory source declares exactly what that loader collects.
    """
    try:
        members = sorted(directory.glob("*.json"))
    except (OSError, ValueError) as error:
        return BROKEN, [], f"{directory}: include source did not load: {error}"
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in members:
        try:
            document = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError) as error:
            return BROKEN, [], f"{entry}: include source did not load: {error}"
        if isinstance(document, dict) and "name" in document:
            declared_name = document["name"]
            if not isinstance(declared_name, str) or not declared_name:
                # The loader keys on the *presence* of `name` and would collect an
                # instance with no name at all; a record has to name itself.
                return BROKEN, [], (
                    f"{entry}: include source did not load: declares an empty or "
                    "non-string `name`, so the loader would collect it nameless"
                )
        name = document.get("name") if isinstance(document, dict) else None
        if not isinstance(name, str) or not name:
            # Not one record: an array or a nameless mapping. The loader skips it,
            # so a selector that asks for an id inside it has to be told which file
            # holds that id instead of hunting a catalogue that looks complete.
            if unreadable is not None:
                for other in _candidate_ids(document):
                    unreadable.setdefault(other, entry.name)
            continue
        declared = document.get("id")
        if isinstance(declared, str) and declared and declared != name:
            return BROKEN, [], (
                f"{entry}: declares id {declared!r} and name {name!r}; "
                "the loader collects by `name`"
            )
        if name in seen:
            return BROKEN, [], (
                f"{entry}: include source did not load: declares duplicate problem id {name!r}"
            )
        seen.add(name)
        records.append({**document, "id": name})
    return DIRECTORY, records, None


def _candidate_ids(document: Any) -> list[str]:
    """Ids a directory member advertises without being one record."""
    if isinstance(document, list):
        rows = document
    elif isinstance(document, dict):
        rows = document.get("cases")
        if rows is None:
            rows = document.get("patterns")
    else:
        rows = None
    if not isinstance(rows, list):
        return []
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("id", "name"):
            value = row.get(key)
            if isinstance(value, str) and value:
                ids.append(value)
                break
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
