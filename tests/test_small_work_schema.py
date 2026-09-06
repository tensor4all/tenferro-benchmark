#!/usr/bin/env python3
"""Regression checks for the common run envelope and small-work extension."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_small_work  # noqa: E402
from small_work import validate_suite_contract, make_record, CaseContract  # noqa: E402


class SmallWorkSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_schema = json.loads((ROOT / "schemas/benchmark-run.schema.json").read_text())
        cls.suite_schema = json.loads((ROOT / "schemas/small-work-suite.schema.json").read_text())
        cls.result_schema = json.loads((ROOT / "schemas/small-work-result.schema.json").read_text())

    def validate(self, schema, value):
        jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(value)

    def metadata(self):
        # Synthetic schema fixture: no dependence on a previous session's /tmp artifacts.
        checkout = {"path": "/fixture/library", "head": "a" * 40, "dirty": False,
                    "untracked": False, "status_sha256": "b" * 64,
                    "tracked_file_count": 1, "tracked_files_sha256": "c" * 64, "error": None}
        binding = {"export": "export.json", "inventory": "inventory.json", "checkout": checkout,
                   "checkout_path": checkout["path"], "checkout_head": checkout["head"],
                   "checkout_revision": checkout["head"], "checkout_dirty": False,
                   "checkout_untracked": False, "checkout_error": None,
                   "export_sha256": "d" * 64, "inventory_sha256": "e" * 64,
                   "contract_ids": ["core.add.ordinary.concrete"],
                   "checker": {"returncode": 0, "stdout": "", "stderr": ""}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.yaml"
            run_small_work._write_frozen_metadata(
                path, args=SimpleNamespace(target_profile="amd-cpu"),
                suite_path=ROOT / "benchmarks/cpu/small_work.yaml",
                timestamp="2026-09-05T15:01:51Z", protocol={},
                selection={"status": "inconclusive", "requested_threads": 1,
                           "effective_threads": 0, "cpus": [], "reasons": ["schema fixture"]},
                provenance={"status": "unverified", "reasons": ["schema fixture"]}, binding=binding)
            return yaml.safe_load(path.read_text())

    def test_manifest_contains_balanced_24_case_matrix(self):
        suite = yaml.safe_load((ROOT / "benchmarks/cpu/small_work.yaml").read_text())
        cases = [case for case in validate_suite_contract(suite) if case.operation == "add"]
        self.assertEqual(len(cases), 24)
        self.assertEqual({case.api_tier for case in cases}, {"concrete-fresh", "concrete-shared", "eager-no-ad", "eager-ad"})
        self.assertEqual({case.shape[0] for case in cases}, {4, 16, 256})
        self.assertEqual({case.workflow for case in cases}, {"single", "dependent10"})
        for tier in {case.api_tier for case in cases}:
            for size in (4, 16, 256):
                selected = [case for case in cases if case.api_tier == tier and case.shape == (size,)]
                self.assertEqual({case.workflow for case in selected}, {"single", "dependent10"})

    def test_einsum_matrix_includes_setup_and_prepared_execution(self):
        suite = yaml.safe_load((ROOT / "benchmarks/cpu/small_work.yaml").read_text())
        cases = [case for case in validate_suite_contract(suite)
                 if case.operation == "einsum" and case.dtype == "f64" and not case.api_tier.startswith("borrowed-") and case.api_tier != "compiled-repeat"]
        self.assertEqual(len(cases), 18)
        for n in (2, 4, 16):
            self.assertEqual({case.api_tier for case in cases if case.shape == (n, n)},
                             {"concrete-fresh", "concrete-shared", "eager-no-ad", "eager-ad", "prepared-setup", "prepared-repeat"})
        for tier, wrong_phase in (("prepared-setup", "execution"), ("prepared-repeat", "setup"), ("compiled-repeat", "setup")):
            case = dict(next(case for case in suite["cases"] if case["api_tier"] == tier))
            case["phase"] = wrong_phase
            with self.assertRaises(ValueError):
                CaseContract.from_mapping(case)

    def test_complex_matrix_preserves_concrete_identity(self):
        suite = yaml.safe_load((ROOT / "benchmarks/cpu/small_work.yaml").read_text())
        cases = [case for case in validate_suite_contract(suite)
                 if case.dtype == "c64" and case.api_tier in {"concrete-fresh", "concrete-shared"}]
        self.assertEqual(len(cases), 6)
        for n in (2, 4, 16):
            selected = [case for case in cases if case.shape == (n, n)]
            self.assertEqual({case.api_tier for case in selected}, {"concrete-fresh", "concrete-shared"})
            for case in selected:
                self.assertEqual(case.contract_id, "einsum.einsum.ordinary.concrete")
                self.assertEqual(case.layout, "col_major_contiguous")
                self.assertEqual(case.workflow, "single")

    def test_complex_prepared_matrix_keeps_setup_separate(self):
        suite = yaml.safe_load((ROOT / "benchmarks/cpu/small_work.yaml").read_text())
        cases = [case for case in validate_suite_contract(suite)
                 if case.dtype == "c64" and case.api_tier.startswith("prepared-")]
        self.assertEqual(len(cases), 6)
        for n in (2, 4, 16):
            selected = [case for case in cases if case.shape == (n, n)]
            self.assertEqual({case.api_tier for case in selected}, {"prepared-setup", "prepared-repeat"})
            for case in selected:
                setup = case.api_tier == "prepared-setup"
                self.assertEqual(case.phase, "setup" if setup else "execution")
                self.assertEqual(case.contract_id, "einsum.einsum.prepare.concrete" if setup else "einsum.einsum.prepared.concrete")
                self.assertEqual(case.scope_timer, ("prepare", "plan_lifetime") if setup else ("prepared_execute", "output_lifetime"))

    def test_compiled_matrix_is_traced_execution_with_setup_outside(self):
        suite = yaml.safe_load((ROOT / "benchmarks/cpu/small_work.yaml").read_text())
        cases = [case for case in validate_suite_contract(suite) if case.api_tier == "compiled-repeat"]
        self.assertEqual(len(cases), 6)
        for dtype in ("f64", "c64"):
            self.assertEqual({case.shape for case in cases if case.dtype == dtype}, {(2, 2), (4, 4), (16, 16)})
        for case in cases:
            self.assertEqual(case.contract_id, "einsum.einsum.prepared.traced")
            self.assertEqual(case.surface, "traced")
            self.assertEqual(case.phase, "execution")
            self.assertIn("trace_compile", case.scope_outside_timer)
            self.assertIn("input_bindings", case.scope_outside_timer)
            self.assertNotIn("trace_compile", case.scope_timer)

    def test_complex_ad_matrix_times_recording_not_backward(self):
        suite = yaml.safe_load((ROOT / "benchmarks/cpu/small_work.yaml").read_text())
        cases = [case for case in validate_suite_contract(suite)
                 if case.dtype == "c64" and case.api_tier == "eager-ad"]
        self.assertEqual(len(cases), 3)
        self.assertEqual({case.shape for case in cases}, {(2, 2), (4, 4), (16, 16)})
        for case in cases:
            self.assertEqual(case.contract_id, "einsum.einsum.ordinary.eager")
            self.assertEqual(case.scope_timer, ("einsum_forward_recording", "output_materialization"))
            self.assertIn("backward", case.scope_outside_timer)

    def test_complex_eager_matrix_does_not_claim_ad(self):
        suite = yaml.safe_load((ROOT / "benchmarks/cpu/small_work.yaml").read_text())
        cases = [case for case in validate_suite_contract(suite)
                 if case.dtype == "c64" and case.api_tier == "eager-no-ad"]
        self.assertEqual(len(cases), 3)
        self.assertEqual({case.shape for case in cases}, {(2, 2), (4, 4), (16, 16)})
        for case in cases:
            self.assertEqual(case.contract_id, "einsum.einsum.ordinary.eager")
            self.assertEqual(case.surface, "eager")
            self.assertEqual(case.scope_timer, ("einsum", "output_materialization"))

    def test_borrowed_layout_matrix_keeps_distinct_cases(self):
        suite = yaml.safe_load((ROOT / "benchmarks/cpu/small_work.yaml").read_text())
        cases = [case for case in validate_suite_contract(suite) if case.api_tier.startswith("borrowed-")]
        self.assertEqual(len(cases), 18)
        self.assertEqual(len({case.case_id for case in cases}), 18)
        for n in (2, 4, 16):
            for tier in ("borrowed-fresh", "borrowed-shared"):
                selected = [case for case in cases if case.shape == (n, n) and case.api_tier == tier]
                self.assertEqual({case.layout for case in selected},
                                 {"col_major_contiguous", "row_major_contiguous", "strided"})
                for case in selected:
                    self.assertEqual(case.contract_id, "einsum.einsum.ordinary.concrete")
                    self.assertEqual(case.workflow, "single")
                    self.assertEqual(case.dtype, "f64")
                    self.assertEqual(case.calls_per_workflow, 1)

    def test_frozen_metadata_and_records_validate(self):
        self.validate(self.run_schema, self.metadata())
        suite = yaml.safe_load((ROOT / "benchmarks/cpu/small_work.yaml").read_text())
        self.validate(self.suite_schema, suite)
        for case in validate_suite_contract(suite):
            record = make_record(case, [], correctness_status="passed", timing_status="inconclusive")
            self.validate(self.result_schema, record)

    def test_write_frozen_metadata_shape(self):
        source = self.metadata()
        args = SimpleNamespace(target_profile="amd-cpu")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.yaml"
            run_small_work._write_frozen_metadata(
                path, args=args, suite_path=ROOT / "benchmarks/cpu/small_work.yaml",
                timestamp="2026-09-05T15:01:51Z", selection=source["small_work"]["resource"],
                protocol=source["small_work"]["protocol"],
                provenance=source["small_work"]["provenance"],
                binding=source["small_work"]["canonical_binding"],
            )
            self.validate(self.run_schema, yaml.safe_load(path.read_text()))

    def test_receipt_is_optional_but_valid_when_present(self):
        metadata = self.metadata()
        metadata["small_work"]["provenance"]["receipt"] = {"file": "receipt.json", "sha256": "a" * 64}
        self.validate(self.run_schema, metadata)
        del metadata["small_work"]["provenance"]["receipt"]
        self.validate(self.run_schema, metadata)

    def test_wrong_contract_and_missing_binding_rejected(self):
        metadata = self.metadata()
        metadata["small_work"]["contract_version"] = 1
        with self.assertRaises(jsonschema.ValidationError):
            self.validate(self.run_schema, metadata)
        metadata["small_work"]["contract_version"] = 2
        del metadata["small_work"]["canonical_binding"]
        with self.assertRaises(jsonschema.ValidationError):
            self.validate(self.run_schema, metadata)


if __name__ == "__main__":
    unittest.main()
