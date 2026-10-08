#!/usr/bin/env python3
"""Public-API entrypoint validation and route coverage accounting."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import route_coverage as rc  # noqa: E402

MANIFEST = yaml.safe_load(rc.MANIFEST.read_text())


class EntrypointTest(unittest.TestCase):
    def test_removed_spellings_are_detected_in_a_synthetic_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "crates/demo/src"
            src.mkdir(parents=True)
            (src / "lib.rs").write_text(
                "pub trait EagerSessionEinsumExt { fn einsum(&mut self); }\n"
                "pub struct TypedTensorView;\n"
                "impl<'a, T> TypedTensorView<'a, T> { pub fn reshape_view(&self) {} }\n")
            index = rc.SourceIndex(Path(tmp))
            manifest = {"coverage": [{"family": "f", "disposition": "measured", "tenferro_apis": [
                "EagerSessionEinsumExt::einsum", "EagerEinsumExt::einsum",
                "TypedTensorView::{reshape_view,try_reshape}", "free text API"]}]}
            problems = rc.check_entrypoints(manifest, index)
            self.assertEqual(problems, ["f: EagerEinsumExt is not defined",
                                        "f: TypedTensorView::try_reshape is not defined",
                                        "f: free-text entrypoint 'free text API' is not a real spelling"])

    @unittest.skipUnless((rc.TENFERRO / "crates").exists(), "extern/tenferro-rs not checked out")
    def test_manifest_names_only_real_entrypoints(self):
        self.assertEqual(rc.check_entrypoints(MANIFEST, rc.SourceIndex(rc.TENFERRO)), [])

    def test_route_cases_exist_and_aliases_claim_no_routes(self):
        cases = {"cpu/session_matrix": {c["id"] for c in json.loads(
                     (ROOT / "data/instances/session_matrix.json").read_text())},
                 "cpu/small_work": {c["id"] for c in json.loads(
                     (ROOT / "data/instances/small_work.json").read_text())}}
        for route in MANIFEST["routes"]:
            for ref in route["cases"]:
                suite, _, case_id = ref.partition(":")
                self.assertIn(case_id.split("[")[0], cases[suite], ref)
        self.assertNotIn("EagerTensor::add", rc.MANIFEST.read_text())
        f2 = {r["route_id"] for r in MANIFEST["routes"]}
        self.assertTrue({"dot_read_batched_alloc", "dot_read_batched_into",
                         "dot_read_loop_of_independent_calls"} <= f2)


class RouteCoverageTest(unittest.TestCase):
    def status(self, tmp, **fields):
        base = {"case_status_version": 1, "suite_id": "cpu/session_matrix", "manifest_version": 3,
                "coverage": "quick", "effort": "scan", "selection_filter": None, "complete": True,
                "expected": [], "selected": [], "not_selected": [], "executed": [],
                "unsupported": [], "failed": [], "missing": [], "noisy": [],
                "unexpected_rows": [], "duplicate_rows": []}
        base.update(fields)
        run = Path(tmp) / "run"
        run.mkdir(exist_ok=True)
        (run / "case_status.json").write_text(json.dumps(base))
        return rc.load_status([run])

    def test_only_executed_cases_cover_a_route(self):
        manifest = {"routes": [
            {"route_id": "a", "entrypoint": "X::a", "cases": ["cpu/session_matrix:c1[faer]"]},
            {"route_id": "b", "entrypoint": "X::b", "cases": ["cpu/session_matrix:c2[faer]"]},
            {"route_id": "c", "entrypoint": "X::c", "cases": ["cpu/session_matrix:c3[faer]"]},
            {"route_id": "d", "entrypoint": "X::d", "cases": ["cpu/session_matrix:c4[faer]"]},
            {"route_id": "e", "entrypoint": "X::e", "cases": ["cpu/session_matrix:c5[faer]"]},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            status = self.status(tmp, executed=["c1[faer]@t1", "c1[faer]@t4"],
                                 unsupported=["c2[faer]@t1"], failed=["c3[faer]@t4"],
                                 missing=["c4[faer]@t1"])
        rows = {r["route_id"]: r for r in rc.route_coverage(manifest, status)}
        self.assertTrue(rows["a"]["covered"])
        for route_id, why in [("b", "unsupported"), ("c", "failed"), ("d", "missing"), ("e", "not run")]:
            self.assertFalse(rows[route_id]["covered"])
            self.assertIn(why, rows[route_id]["reasons"][0])


if __name__ == "__main__":
    unittest.main()
