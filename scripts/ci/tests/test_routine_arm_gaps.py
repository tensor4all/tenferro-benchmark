"""A narrowed run must not replace a routine page."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ci"))

import yaml

from scripts.ci.check_gpu_run import routine_arm_gaps, routine_arms
from scripts.ci.ingest_benchmark_results import refuse_narrowed_routine_arms


class RoutineArmGapTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)

    def plan(self, pairs):
        (self.run / "run.yaml").write_text(yaml.safe_dump(
            {"collection": {"expected_case_ids": [f"{suite}:{case}:{backend}"
                                                  for suite, case, backend in pairs]}}))

    def test_the_manifest_declares_the_arms_a_routine_page_needs(self):
        self.assertIn("jax-cuda", routine_arms()["gpu/dense"])
        # The tensor-network routine arm is the tree JIT path, not plain jax-cuda.
        self.assertIn("jax-cuda-tree-jit", routine_arms()["gpu/tensornetwork"])

    def test_a_plan_that_dropped_an_arm_reports_it(self):
        self.plan([("gpu/dense", "matmul", arm) for arm in
                   ("tenferro-cuda-trace", "tenferro-cuda-eager", "pytorch-cuda")])
        self.assertEqual(routine_arm_gaps(self.run, "gpu/dense"), ["jax-cuda"])

    def test_a_complete_plan_reports_nothing(self):
        arms = ("tenferro-cuda-trace", "tenferro-cuda-eager", "pytorch-cuda", "jax-cuda")
        self.plan([("gpu/dense", "matmul", arm) for arm in arms])
        self.assertEqual(routine_arm_gaps(self.run, "gpu/dense"), [])

    def test_a_run_without_a_plan_claims_no_gap(self):
        (self.run / "run.yaml").write_text("suite_id: gpu/dense\n")
        self.assertEqual(routine_arm_gaps(self.run, "gpu/dense"), [])

    def test_a_suite_outside_the_manifest_needs_nothing(self):
        self.plan([("gpu/dense", "matmul", "pytorch-cuda")])
        self.assertEqual(routine_arm_gaps(self.run, "gpu/not_routine"), [])


class RefuseNarrowedPublicationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bundle = Path(self.temporary.name)
        self.publication = {"schema_version": 2, "timestamp": "20261108_090000", "suites": []}
        self.write()

    def write(self):
        (self.bundle / "publication.json").write_text(json.dumps(self.publication))

    def test_a_narrowed_run_is_refused_with_the_missing_arms_named(self):
        self.publication["narrowed_routine_arms"] = {"gpu/dense": ["jax-cuda"]}
        self.write()
        with self.assertRaisesRegex(ValueError, "jax-cuda"):
            refuse_narrowed_routine_arms(self.bundle, "nvidia-gpu")

    def test_a_complete_run_publishes(self):
        refuse_narrowed_routine_arms(self.bundle, "nvidia-gpu")

    def test_a_cpu_profile_is_not_checked(self):
        self.publication["narrowed_routine_arms"] = {"gpu/dense": ["jax-cuda"]}
        self.write()
        refuse_narrowed_routine_arms(self.bundle, "mac-cpu")


if __name__ == "__main__":
    unittest.main()
