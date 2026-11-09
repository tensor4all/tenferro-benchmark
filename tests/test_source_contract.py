"""One table over every source shape a suite selector can point at.

"What is a source" used to be decided in three places -- the validator's
`regime_policy.check_suite`, the resolver's `suite_instances.resolve_suite_instance_ids`
and the collector's `benchmark_python.load_instances` -- and each read the same file
its own way.  Four review rounds produced a fix on one side and a gap on another:

- a `cases.json` with a trailing comma was `False` to the validator (which parses
  JSON) and `["a"]` to the resolver (which fell back to YAML, where the comma is
  legal);
- a `{"name": ..., "patterns": [...]}` catalogue validated, then collection died on
  the id it declared;
- one `{"name": "a"}` record was collected by the name-keyed loader and rejected by
  the validator.

`suite_instances.resolve_source()` is now the only classifier, and the validator and
the resolver both call it.  Every row below builds one source shape, then calls all
three paths for real and asserts that they agree: the same verdict, the same id set,
and -- where the name-keyed loader is the collector for that shape -- the same
records.  A new shape needs a row here, and `SOURCE_KINDS` says when one is missing.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402

import benchmark_python  # noqa: E402
import regime_policy  # noqa: E402
import suite_instances  # noqa: E402
import validate_benchmark_suite as validator  # noqa: E402

SUITE_ID = "cpu/einsum"
LATENCY = {"regime": "latency"}


def wrapper(document: dict, include: list[str]) -> dict:
    """A schema-valid suite that includes `include` from `document`."""
    return {
        "schema_version": 1,
        "suite_id": SUITE_ID,
        "title": "source contract fixture",
        "description": "source contract fixture",
        "defaults": LATENCY,
        "backends": ["tenferro-eager"],
        "problems": {"source": str(document), "include": include},
    }


def nested_suite(case_ids: list[str]) -> dict:
    """A schema-valid suite document, used as somebody else's `source`."""
    return {
        "schema_version": 1,
        "suite_id": "cpu/base",
        "title": "nested fixture",
        "description": "nested fixture",
        "defaults": LATENCY,
        "backends": ["tenferro-eager"],
        "problems": [{"id": case_id, **LATENCY} for case_id in case_ids],
    }


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The table.  Each `write` takes the collector's data directory and the fixture
# root, and returns the `source` value plus the `include` list the suite declares.
# `kind` is what `resolve_source()` must report, `outcome` how the selector ends,
# `ids` the agreed id set when it works, and `collected` the instance names the
# name-keyed loader reads, so the collector's own reading is pinned too.
# ---------------------------------------------------------------------------


def directory_of_records(data: Path, tmp: Path):
    for name in ("a", "b"):
        write_json(data / "instances" / f"{name}.json", {"name": name, **LATENCY})
    return data / "instances", ["a", "b"]


def directory_with_an_array(data: Path, tmp: Path):
    write_json(data / "instances" / "a.json", {"name": "a", **LATENCY})
    write_json(data / "instances" / "group.json", [{"name": "b", **LATENCY}])
    return data / "instances", ["a", "b"]


def cases_catalogue(data: Path, tmp: Path):
    path = write_json(
        data / "cases.json",
        {"cases": [{"id": "a", **LATENCY}, {"id": "b", **LATENCY}]},
    )
    return path, ["a", "b"]


def patterns_catalogue(data: Path, tmp: Path):
    path = write_json(
        data / "patterns.json",
        {"version": 1, "patterns": [{"id": "a", **LATENCY}, {"id": "b", **LATENCY}]},
    )
    return path, ["a", "b"]


def patterns_catalogue_that_names_itself(data: Path, tmp: Path):
    path = write_json(
        data / "catalogue.json",
        {"name": "catalogue", "patterns": [{"id": "a", **LATENCY}]},
    )
    return path, ["a"]


def bare_array_catalogue(data: Path, tmp: Path):
    path = write_json(data / "records.json", [{"id": "a", **LATENCY}, {"id": "b", **LATENCY}])
    return path, ["a", "b"]


def single_instance_record(data: Path, tmp: Path):
    return write_json(data / "a.json", {"name": "a", **LATENCY}), ["a"]


def malformed_json(data: Path, tmp: Path):
    path = data / "cases.json"
    path.write_text('{"cases":[{"id":"a","regime":"latency",}],}', encoding="utf-8")
    return path, ["a"]


def yaml_catalogue(data: Path, tmp: Path):
    path = data / "cases.yaml"
    path.write_text(
        "cases:\n  - id: a\n    regime: latency\n  - id: b\n    regime: latency\n",
        encoding="utf-8",
    )
    return path, ["a", "b"]


def json_syntax_suite(data: Path, tmp: Path):
    path = data / "base.yaml"
    path.write_text(json.dumps(nested_suite(["a"])), encoding="utf-8")
    return path, ["a"]


def yaml_suite(data: Path, tmp: Path):
    path = data / "base.yaml"
    path.write_text(yaml.safe_dump(nested_suite(["a"])), encoding="utf-8")
    return path, ["a"]


def unknown_include_id(data: Path, tmp: Path):
    path = write_json(data / "cases.json", {"cases": [{"id": "a", **LATENCY}]})
    return path, ["a", "b"]


def repeated_include_id(data: Path, tmp: Path):
    path = write_json(data / "cases.json", {"cases": [{"id": "a", **LATENCY}]})
    return path, ["a", "a"]


def suite_and_catalogue(data: Path, tmp: Path):
    path = write_json(
        data / "both.json",
        {"problems": [{"id": "a", **LATENCY}], "cases": [{"id": "b", **LATENCY}]},
    )
    return path, ["a"]


def nested_suite_with_an_unknown_include_id(data: Path, tmp: Path):
    path = data / "base.yaml"
    path.write_text(yaml.safe_dump(nested_suite(["a"])), encoding="utf-8")
    return path, ["a", "b"]


def directory_with_a_nameless_cases_member(data: Path, tmp: Path):
    write_json(data / "instances" / "a.json", {"name": "a", **LATENCY})
    write_json(data / "instances" / "part.json", {"cases": [{"id": "b", **LATENCY}]})
    return data / "instances", ["a", "b"]


def directory_with_a_nameless_patterns_member(data: Path, tmp: Path):
    write_json(data / "instances" / "a.json", {"name": "a", **LATENCY})
    write_json(
        data / "instances" / "patterns.json", {"patterns": [{"id": "b", **LATENCY}]}
    )
    return data / "instances", ["a", "b"]


def directory_with_a_nameless_record(data: Path, tmp: Path):
    write_json(data / "instances" / "a.json", {"name": "a", **LATENCY})
    write_json(data / "instances" / "empty.json", {"name": "", **LATENCY})
    return data / "instances", ["a"]


def self_including_suite(data: Path, tmp: Path):
    path = data / "base.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **nested_suite([]),
                "problems": {"source": str(path), "include": ["a"]},
            }
        ),
        encoding="utf-8",
    )
    return path, ["a"]


def json_file_holding_yaml(data: Path, tmp: Path):
    path = data / "yaml_in_json.json"
    path.write_text("cases:\n  - id: a\n    regime: latency\n", encoding="utf-8")
    return path, ["a"]


def yaml_whose_resolver_fails(data: Path, tmp: Path):
    # PyYAML 6.0.3's implicit `!!timestamp` resolver raises `AttributeError` -- not
    # `YAMLError` -- for an unparsable value, so this source used to escape the
    # per-source boundary and abort the validator's remaining inputs.
    (data / "instances").mkdir(parents=True, exist_ok=True)
    path = data / "bad.yaml"
    path.write_text(
        "cases:\n  - id: a\n    regime: latency\n    date: !!timestamp not-a-date\n",
        encoding="utf-8",
    )
    return path, ["a"]


TABLE = [
    {
        "shape": "a directory of name records",
        "write": directory_of_records,
        "kind": suite_instances.DIRECTORY,
        "outcome": "ok",
        "ids": ["a", "b"],
        "collected": ["a", "b"],
        "collect_in": "instances",
    },
    {
        "shape": "a directory with an array member",
        "write": directory_with_an_array,
        "kind": suite_instances.DIRECTORY,
        "outcome": "error",
        "ids": [],
        "collected": ["a"],
        "collect_in": "instances",
        "names": ["group.json", "'b'"],
    },
    {
        "shape": "a cases catalogue",
        "write": cases_catalogue,
        "kind": suite_instances.CATALOGUE_CASES,
        "outcome": "ok",
        "ids": ["a", "b"],
        # A catalogue is not one record: the name-keyed loader skips it, exactly as
        # it skips `data/instances/perf_issues.json`.
        "collected": [],
    },
    {
        "shape": "a patterns catalogue",
        "write": patterns_catalogue,
        "kind": suite_instances.CATALOGUE_PATTERNS,
        "outcome": "ok",
        "ids": ["a", "b"],
        "collected": [],
    },
    {
        "shape": "a patterns catalogue that names itself",
        "write": patterns_catalogue_that_names_itself,
        "kind": suite_instances.CATALOGUE_PATTERNS,
        "outcome": "ok",
        # `patterns` wins over `name`: a catalogue stays a catalogue. It is still one
        # record to the name-keyed loader, which is the collector's own business and
        # why this row pins both readings instead of hiding one.
        "ids": ["a"],
        "collected": ["catalogue"],
    },
    {
        "shape": "a bare array catalogue",
        "write": bare_array_catalogue,
        "kind": suite_instances.CATALOGUE_CASES,
        "outcome": "ok",
        "ids": ["a", "b"],
        "collected": [],
    },
    {
        "shape": "one instance record",
        "write": single_instance_record,
        "kind": suite_instances.INSTANCE_RECORD,
        "outcome": "ok",
        "ids": ["a"],
        "collected": ["a"],
    },
    {
        "shape": "a YAML catalogue",
        "write": yaml_catalogue,
        "kind": suite_instances.CATALOGUE_CASES,
        "outcome": "ok",
        "ids": ["a", "b"],
        "collected": None,  # `.yaml`: the name-keyed loader globs `*.json` only
    },
    {
        "shape": "a JSON-syntax suite",
        "write": json_syntax_suite,
        "kind": suite_instances.SUITE_DOCUMENT,
        "outcome": "ok",
        "ids": ["a"],
        "collected": None,  # `.yaml`: the name-keyed loader globs `*.json` only
    },
    {
        "shape": "a YAML suite",
        "write": yaml_suite,
        "kind": suite_instances.SUITE_DOCUMENT,
        "outcome": "ok",
        "ids": ["a"],
        "collected": None,  # `.yaml`: the name-keyed loader globs `*.json` only
    },
    {
        "shape": "a nested suite with an unknown include id",
        "write": nested_suite_with_an_unknown_include_id,
        # A nested suite is not a catalogue, but an id it does not declare is the
        # same fail-open: the validator rejected it while the resolver kept the
        # subset it recognised.
        "kind": suite_instances.SUITE_DOCUMENT,
        "outcome": "error",
        "ids": [],
        "collected": None,
        "names": ["'b'"],
    },
    {
        "shape": "a directory with a nameless cases member",
        "write": directory_with_a_nameless_cases_member,
        "kind": suite_instances.DIRECTORY,
        "outcome": "error",
        "ids": [],
        "collected": ["a"],
        "collect_in": "instances",
        "names": ["part.json", "'b'"],
    },
    {
        "shape": "a directory with a nameless patterns member",
        "write": directory_with_a_nameless_patterns_member,
        "kind": suite_instances.DIRECTORY,
        "outcome": "error",
        "ids": [],
        "collected": ["a"],
        "collect_in": "instances",
        "names": ["patterns.json", "'b'"],
    },
    {
        "shape": "a directory member that names itself with an empty name",
        "write": directory_with_a_nameless_record,
        # One member that cannot name itself makes the whole directory broken: the
        # loader keys on the *presence* of `name` and would collect it nameless, so
        # skipping it quietly hides a record the run would still pick up.
        "kind": suite_instances.BROKEN,
        "outcome": "error",
        "ids": [],
        "collected": ["a", ""],
        "collect_in": "instances",
        "names": ["empty.json", "`name`"],
        "error_files": ["empty.json"],
    },
    {
        "shape": "a suite that includes itself",
        "write": self_including_suite,
        "kind": suite_instances.SUITE_DOCUMENT,
        "outcome": "error",
        "ids": [],
        "collected": None,  # `.yaml`: the name-keyed loader globs `*.json` only
        # A chain that includes itself ends in `RecursionError` inside the parser,
        # which `resolve_source()` reports as a broken source naming the file: the
        # validator and the resolver used to disagree on whether it loaded at all.
        "names": ["maximum recursion depth exceeded"],
    },
    {
        "shape": "malformed JSON in a .json source",
        "write": malformed_json,
        "kind": suite_instances.BROKEN,
        "outcome": "error",
        "ids": [],
        "collected": [],
        "names": ["cases.json"],
    },
    {
        "shape": "a .json source holding YAML",
        "write": json_file_holding_yaml,
        "kind": suite_instances.BROKEN,
        "outcome": "error",
        "ids": [],
        "collected": [],
        "names": ["yaml_in_json.json"],
    },
    {
        "shape": "an unknown include id",
        "write": unknown_include_id,
        "kind": suite_instances.CATALOGUE_CASES,
        "outcome": "error",
        "ids": [],
        "collected": [],
        "names": ["'b'"],
    },
    {
        "shape": "an include that repeats an id",
        "write": repeated_include_id,
        "kind": suite_instances.CATALOGUE_CASES,
        "outcome": "error",
        "ids": [],
        "collected": [],
        "names": ["repeats id 'a'"],
        "validator_names": ["is not valid under any of the given schemas"],
    },
    {
        "shape": "a document that is a suite and a catalogue",
        "write": suite_and_catalogue,
        "kind": suite_instances.BROKEN,
        "outcome": "error",
        "ids": [],
        "collected": [],
        "names": ["both.json", "`problems` and `cases`"],
    },
    {
        "shape": "a YAML source whose parser raises outside `YAMLError`",
        "write": yaml_whose_resolver_fails,
        "kind": suite_instances.BROKEN,
        "outcome": "error",
        "ids": [],
        "collected": [],
        "collect_in": "instances",
        "names": ["bad.yaml", "include source did not load"],
    },
]


class SourceContractTests(unittest.TestCase):
    def run_row(self, row: dict) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            data = tmp / "data"
            data.mkdir()
            source, include = row["write"](data, tmp)
            suite = wrapper(source, include)
            suite_file = tmp / "wrapper.yaml"
            suite_file.write_text(yaml.safe_dump(suite), encoding="utf-8")

            kind, records, error = suite_instances.resolve_source(Path(source))
            self.assertEqual(kind, row["kind"], row["shape"])
            for token in row.get("names", []):
                if kind == suite_instances.BROKEN:
                    self.assertIn(token, error or "", row["shape"])
                    for name in row.get("error_files", [str(source)]):
                        self.assertIn(name, error or "", row["shape"])

            # 3. the collector: the same loader the run uses, on the same directory.
            # `BENCH_SUITE_INCLUDE` filters it from the environment, so a row's
            # expectation has to be read without one.
            collect_dir = data / row.get("collect_in", "")
            original = benchmark_python.DATA_DIR
            benchmark_python.DATA_DIR = collect_dir
            try:
                with mock.patch.dict(os.environ, {"BENCH_SUITE_INCLUDE": ""}):
                    with contextlib.redirect_stderr(io.StringIO()):
                        collected = [
                            record["name"] for record in benchmark_python.load_instances()
                        ]
            finally:
                benchmark_python.DATA_DIR = original

            # 1. the validator, as `validate_benchmark_suite.py` runs it.
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                validator_ok = validator.validate_suite(suite_file)
            validator_message = stderr.getvalue()

            # 2. the resolver `run_all.sh` collects through.
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    resolved, resolver_message = (
                        suite_instances.resolve_suite_instance_ids(suite_file),
                        "",
                    )
                except ValueError as raised:
                    resolved, resolver_message = None, str(raised)

            self.assertEqual(validator_ok, resolved is not None, row["shape"])
            if row["outcome"] == "ok":
                self.assertTrue(validator_ok, validator_message)
                declared = [case[0] for case in regime_policy.suite_cases(suite, tmp)]
                self.assertEqual(resolved, row["ids"], row["shape"])
                self.assertEqual(declared, row["ids"], row["shape"])
            else:
                self.assertFalse(validator_ok, row["shape"])
                # The validator rejects a repeated include id in the schema before
                # the regime check runs, so a row may pin the two messages apart.
                for token in row.get("validator_names", row["names"]):
                    self.assertIn(token, validator_message, row["shape"])
                for token in row["names"]:
                    self.assertIn(token, resolver_message, row["shape"])

            if row["collected"] is not None:
                self.assertEqual(collected, row["collected"], row["shape"])
            if row["outcome"] == "ok" and kind in (
                suite_instances.DIRECTORY,
                suite_instances.INSTANCE_RECORD,
            ):
                # The name-keyed loader is the collector for these shapes: the
                # records the classifier reports and the records it reads are the
                # same ones, in the same order.
                self.assertEqual([record["id"] for record in records], collected, row["shape"])

    def test_every_source_shape_is_read_the_same_way_three_times(self):
        self.assertEqual(
            len(TABLE),
            21,
            "one row per source shape: a new shape needs a row here",
        )
        for row in TABLE:
            with self.subTest(shape=row["shape"]):
                self.run_row(row)

    def test_the_table_covers_every_source_kind(self):
        # A kind `resolve_source()` can return but no row exercises is a shape whose
        # three readings nobody checked.
        self.assertEqual(
            {row["kind"] for row in TABLE},
            set(suite_instances.SOURCE_KINDS),
            "add a row for the kind that is missing from the table",
        )

    def test_a_malformed_input_is_reported_and_the_next_file_still_runs(self):
        # PyYAML 6.0.3 raises `AttributeError` -- not `YAMLError` -- from its
        # `!!timestamp` resolver, which used to abort the command-line inputs after
        # the first bad file.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            bad = tmp / "bad.yaml"
            bad.write_text("date: !!timestamp not-a-date\n", encoding="utf-8")
            cases = write_json(tmp / "cases.json", {"cases": [{"id": "a", **LATENCY}]})
            good = tmp / "good.yaml"
            good.write_text(yaml.safe_dump(wrapper(cases, ["a"])), encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertFalse(validator.validate_suite(bad), stderr.getvalue())
                self.assertTrue(validator.validate_suite(good), stderr.getvalue())
            self.assertIn(str(bad), stderr.getvalue())

    def test_a_malformed_nested_selector_names_the_source_and_the_run_continues(self):
        # A nested suite is resolved without passing through the suite schema, so a
        # wrongly shaped selector reaches `set(problems["exclude"])` as it is; the
        # `TypeError` used to escape the per-source boundary.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            nested = tmp / "nested.yaml"
            nested.write_text(
                yaml.safe_dump(
                    {"problems": {"source": "generated", "include": ["a"], "exclude": [{}]}}
                ),
                encoding="utf-8",
            )
            first = tmp / "first.yaml"
            first.write_text(yaml.safe_dump(wrapper(nested, ["a"])), encoding="utf-8")
            cases = write_json(tmp / "cases.json", {"cases": [{"id": "a", **LATENCY}]})
            second = tmp / "second.yaml"
            second.write_text(yaml.safe_dump(wrapper(cases, ["a"])), encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertFalse(validator.validate_suite(first), stderr.getvalue())
                self.assertTrue(validator.validate_suite(second), stderr.getvalue())
            self.assertIn("nested.yaml", stderr.getvalue())
            self.assertIn("TypeError", stderr.getvalue())


    def test_a_nested_inline_problem_that_is_not_a_mapping_is_refused_by_both_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            nested = tmp / "nested.yaml"
            nested.write_text(
                yaml.safe_dump({"problems": [False, {"id": "a", **LATENCY}]}),
                encoding="utf-8",
            )
            wrapper_file = tmp / "wrapper.yaml"
            wrapper_file.write_text(yaml.safe_dump(wrapper(nested, ["a"])), encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertFalse(validator.validate_suite(wrapper_file), stderr.getvalue())
            self.assertIn("problems list entries must be mappings", stderr.getvalue())
            with self.assertRaises(ValueError) as raised:
                suite_instances.resolve_suite_instance_ids(wrapper_file)
            self.assertIn("problems list entries must be mappings", str(raised.exception))

    def test_a_nested_selector_that_skips_its_source_reads_the_same_ids_in_both_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            nested = tmp / "nested.yaml"
            nested.write_text(
                yaml.safe_dump(
                    {
                        "problems": {
                            "source": str(tmp / "absent.json"),
                            "include": ["a"],
                            "missing_source": "skip",
                        }
                    }
                ),
                encoding="utf-8",
            )
            wrapper_file = tmp / "wrapper.yaml"
            wrapper_file.write_text(yaml.safe_dump(wrapper(nested, ["a"])), encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertTrue(validator.validate_suite(wrapper_file), stderr.getvalue())
            self.assertIn("missing_source: skip", stderr.getvalue())
            self.assertEqual(
                suite_instances.resolve_suite_instance_ids(wrapper_file), ["a"]
            )

    def test_a_nested_include_entry_that_is_not_an_id_is_refused_by_both_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cases = write_json(tmp / "cases.json", {"cases": [{"id": "a", **LATENCY}]})
            nested = tmp / "nested.yaml"
            nested.write_text(
                yaml.safe_dump({"problems": {"source": str(cases), "include": ["a", 7]}}),
                encoding="utf-8",
            )
            wrapper_file = tmp / "wrapper.yaml"
            wrapper_file.write_text(yaml.safe_dump(wrapper(nested, ["a"])), encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertFalse(validator.validate_suite(wrapper_file), stderr.getvalue())
            self.assertIn("include id 7 is not in the source", stderr.getvalue())
            with self.assertRaises(ValueError) as raised:
                suite_instances.resolve_suite_instance_ids(wrapper_file)
            self.assertIn("include id 7 is not in the source", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
