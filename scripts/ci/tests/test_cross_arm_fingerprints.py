import json
import unittest

from scripts.ci.cross_arm_fingerprints import compare_outputs, compare_records, resolve


def block(**overrides):
    value = {
        "sum_re": 10.0, "sum_im": 0.0, "sum_abs": 10.0, "sum_squares": 30.0,
        "max_abs": 4.0, "dot_seeded_re": -1.5, "dot_seeded_im": 0.0, "elements": 4,
    }
    value.update(overrides)
    return value


def record(backend, fingerprints, problem="case", suite="gpu/unit"):
    return {
        "suite_id": suite, "problem_id": problem, "backend": backend,
        "execution": {"notes": json.dumps({"fingerprints": fingerprints})},
    }


class CrossArmComparisonTests(unittest.TestCase):
    def test_identical_arms_agree(self):
        self.assertEqual(compare_outputs(block(), block()), [])

    def test_a_permutation_is_reported_by_the_seeded_dot(self):
        # Same sums, different seeded projection: exactly the case sums cannot see.
        permuted = block(dot_seeded_re=1.5)
        problems = compare_outputs(block(), permuted)
        self.assertEqual(len(problems), 1)
        self.assertIn("dot_seeded_re", problems[0])

    def test_a_scale_difference_is_reported(self):
        problems = compare_outputs(block(), block(sum_squares=31.0))
        self.assertTrue(any("sum_squares" in problem for problem in problems))

    def test_a_cancelling_sum_is_compared_against_the_l1_mass(self):
        # 1e-12 of absolute noise on a sum whose terms cancel is not a disagreement.
        noisy = block(sum_re=10.0 + 1e-12)
        self.assertEqual(compare_outputs(block(), noisy), [])

    def test_a_different_output_count_is_a_disagreement(self):
        report = compare_records([record("a", [block(), block()]), record("b", [block()])])
        self.assertEqual(len(report["disagreements"]), 1)
        self.assertIsNone(report["disagreements"][0]["output"])

    def test_rows_without_invariants_are_skipped_not_failed(self):
        report = compare_records([
            record("rust-arm", None) if False else {
                "suite_id": "gpu/unit", "problem_id": "case", "backend": "rust-arm",
                "execution": {"notes": json.dumps({"fingerprints": []})},
            },
            record("jax", [block()]),
        ])
        self.assertEqual(report["disagreements"], [])
        self.assertEqual(report["single_arm_cases"][0]["arms"], ["jax"])

    def test_a_deferred_row_passes_when_the_other_arm_agrees(self):
        deferred = record("jax", [block()])
        deferred["verification"] = {"status": "skipped",
                                    "reference_backend": "cross_arm_fingerprint"}
        records, report = resolve([deferred, record("pytorch", [block()])])
        self.assertEqual(records[0]["verification"]["status"], "passed")
        self.assertEqual(report["resolved_rows"], 1)
        self.assertEqual(report["unresolved"], [])

    def test_a_deferred_row_fails_when_the_arms_disagree(self):
        deferred = record("jax", [block(dot_seeded_re=1.5)])
        deferred["verification"] = {"status": "skipped",
                                    "reference_backend": "cross_arm_fingerprint"}
        records, report = resolve([deferred, record("pytorch", [block()])])
        self.assertEqual(records[0]["verification"]["status"], "failed")
        self.assertEqual(records[0]["status"], "verification_failed")

    def test_a_deferred_row_with_no_other_arm_stays_unresolved(self):
        deferred = record("jax", [block()])
        deferred["verification"] = {"status": "skipped",
                                    "reference_backend": "cross_arm_fingerprint"}
        records, report = resolve([deferred])
        self.assertEqual(records[0]["verification"]["status"], "skipped")
        self.assertEqual(report["resolved_rows"], 0)
        self.assertEqual(len(report["unresolved"]), 1)

    def test_every_arm_pair_is_compared_once(self):
        report = compare_records([
            record("a", [block()]), record("b", [block()]), record("c", [block()]),
        ])
        self.assertEqual(report["compared_output_pairs"], 3)
        self.assertEqual(report["disagreements"], [])


if __name__ == "__main__":
    unittest.main()
