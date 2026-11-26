"""The tenferro-rs perf issue numbers recorded on cases that already exist.

AGENTS.md requires a tenferro-rs performance issue to become a suite case when
it is opened. When another suite already measures the workload, the record is a
`perf_issues` array on that case's instance entry; the older `issue` field is
the suite's originating tenferro-benchmark issue and must not be overwritten.
`cpu/perf_issues` records the number in the generator's `issues` instead, which
also reaches `data/instances/perf_issues.json`.
"""

import json
import unittest
from pathlib import Path

from scripts import generate_perf_issue_cases as gen

ROOT = Path(__file__).resolve().parents[1]

EXISTING = {
    "data/instances/gpu_nn.json": {
        "softplus_f32_n1048576": [2530],
        "softplus_f64_n1048576": [2530],
    },
    "data/instances/gpu_isolated_ops.json": {
        "reduce_sum_f64_n1048576": [2625],
        "reduce_prod_f64_n1048576": [2625],
        "reduce_max_f64_n1048576": [2625],
        "reduce_min_f64_n1048576": [2625],
    },
    "data/instances/gpu_linalg_public_api.json": {
        "norm_f64_512x512": [2625],
    },
    "data/instances/gpu_permutation_patterns.json": {
        "reverse_18d_3": [2528],
        "reverse_23d_128x2": [2528],
    },
}


def entries(path):
    document = json.loads((ROOT / path).read_text())
    if isinstance(document, list):
        return {case["id"]: case for case in document}
    return {case["id"]: case for case in document.get("cases", document.get("patterns"))}


class PerfIssueCoverageTest(unittest.TestCase):
    def test_existing_cases_carry_the_tracked_issue_number(self):
        for path, expected in EXISTING.items():
            with self.subTest(path=path):
                found = entries(path)
                for case_id, issues in expected.items():
                    self.assertIn(case_id, found)
                    self.assertEqual(
                        sorted(found[case_id].get("perf_issues", [])), sorted(issues)
                    )

    def test_thin_complex_records_2524_and_its_four_thread_control(self):
        rows = gen.thin_complex_cases()
        for row in rows:
            self.assertIn("#2524", row["issues"])
        controls = {row["id"]: row for row in rows if row["thread_counts"] == [4]}
        self.assertEqual(
            set(controls),
            {
                "thin_c64_gemm_8x141_141x282_t4",
                "thin_c64_gemm_8x141_141x282_t4_pytorch",
            },
        )
        cases = entries("data/instances/perf_issues.json")
        recorded = {case_id for case_id, case in cases.items() if "#2524" in case["issues"]}
        self.assertEqual(recorded, {row["id"] for row in rows})
