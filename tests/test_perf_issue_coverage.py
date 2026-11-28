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
        "add_f64_n1048576": [2635],
        "compare_f64_n1048576": [2635],
        "where_select_f64_n1048576": [2635],
        "mul_f64_n1048576": [2635],
        "abs_f64_n1048576": [2635],
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

    def test_small_batched_matmul_pairs_every_cell_with_a_direct_loop_reference(self):
        rows = gen.small_batched_matmul_cases()
        self.assertEqual(len(rows), 6 * 3 * 2 * 2 * 2)
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(len(by_id), len(rows))
        for row in rows:
            self.assertEqual(row["kind"], "small_batched_matmul")
            self.assertIn("#2745", row["issues"])
            self.assertEqual(row["thread_counts"], [1])
            p = row["params"]
            # The label alone names scalar type, size, batch and update mode.
            self.assertIn(f"_{p['dtype']}_{p['n']}x{p['n']}_b{p['batch']}_{p['mode']}", row["id"])
            if row["arm"] == "direct-loop":
                self.assertEqual(row["backend"], "rust-loop")
                self.assertIn(row["reference_for"], by_id)
                self.assertEqual(by_id[row["reference_for"]]["arm"], "tenferro")
            else:
                self.assertEqual(row["backend"], "tenferro-rs")
                self.assertIsNone(row.get("reference_for"))
            # Calls below 2**16 scalars per operand are per-call-cost bound; larger are not.
            elements = p["n"] * p["n"] * p["batch"]
            self.assertEqual(row["size_elements"], elements)
            self.assertEqual(row["regime"], "throughput" if elements >= 1 << 16 else "latency")
        self.assertEqual({row["regime"] for row in rows}, {"latency", "throughput"})
        self.assertEqual({row["params"]["n"] for row in rows}, {1, 2, 3, 4, 5, 8})
        self.assertEqual({row["params"]["batch"] for row in rows}, {256, 4096, 32769})
        self.assertEqual({row["params"]["dtype"] for row in rows}, {"c64", "f64"})
        self.assertEqual({row["params"]["mode"] for row in rows}, {"overwrite", "accumulate"})
        cases = entries("data/instances/perf_issues.json")
        self.assertEqual(
            {case_id for case_id, case in cases.items() if "#2745" in case["issues"]},
            set(by_id),
        )

    def test_ad_prepare_cases_are_diagnostic_and_paired(self):
        rows = (
            gen.ad_prepare_cases()
            + gen.deep_reverse_mode_prepare_cases()
            + gen.small_tn_ad_prepare_cases()
        )
        self.assertEqual(len(rows), 28)
        for row in rows:
            self.assertEqual(row["kind"], "ad_prepare")
            self.assertEqual(row["measurement_scope"], "issue_workflow_diagnostic")
            self.assertEqual(row["regime"], "latency")
            self.assertEqual(row["thread_counts"], [1])
        by_id = {row["id"]: row for row in rows}
        for graph in ("identity_layout_chain", "conj_chain"):
            for length in (64, 256, 1024):
                target = f"{graph}_prepare_c64_len{length}"
                self.assertIn("#2678", by_id[target]["issues"])
                self.assertEqual(by_id[target]["arm"], "vjp")
                self.assertIsNone(by_id[target].get("reference_for"))
                control = by_id[target + "_forward"]
                self.assertEqual(control["arm"], "forward")
                self.assertEqual(control["reference_for"], target)
        for steps in (1, 2, 3, 10):
            target = f"deep_reverse_mode_prepare_f64_steps{steps}"
            self.assertIn("#2643", by_id[target]["issues"])
            self.assertEqual(by_id[target]["arm"], "vjp")
            self.assertEqual(by_id[target]["params"]["batch"], 64)
            self.assertEqual(by_id[target]["params"]["hidden"], 50)
            control = by_id[target + "_forward"]
            self.assertEqual(control["reference_for"], target)
        for sites in (4, 8, 16, 32):
            target = f"small_tn_ad_prepare_c64_sites{sites}"
            vjp = by_id[target]
            self.assertIn("#2678", vjp["issues"])
            self.assertEqual(vjp["arm"], "vjp")
            self.assertEqual(vjp["params"]["graph"], "small_tn")
            self.assertEqual(vjp["params"]["sites"], sites)
            self.assertEqual(vjp["params"]["d"], 2)
            self.assertEqual(vjp["params"]["bond"], 2)
            self.assertIsNone(vjp.get("reference_for"))
            control = by_id[target + "_forward"]
            self.assertEqual(control["arm"], "forward")
            self.assertEqual(control["reference_for"], target)
        cases = entries("data/instances/perf_issues.json")
        for issue, generated in (
            ("#2678", [row["id"] for row in gen.ad_prepare_cases()]
             + [row["id"] for row in gen.small_tn_ad_prepare_cases()]),
            ("#2643", [row["id"] for row in gen.deep_reverse_mode_prepare_cases()]),
        ):
            self.assertEqual(
                {case_id for case_id, case in cases.items() if issue in case["issues"]},
                set(generated),
            )

    def test_matrix_norm_and_rank_gated_reduction_cases(self):
        norm_rows = gen.gpu_matrix_norm_cases(("nvidia-gpu",))
        self.assertEqual(
            {row["id"] for row in norm_rows},
            {
                "matrix_norm_one_f64_n512",
                "matrix_norm_frobenius_f64_n512",
                "matrix_norm_one_f64_n1024",
                "matrix_norm_inf_f64_n1024",
                "matrix_norm_frobenius_f64_n1024",
            },
        )
        for row in norm_rows:
            self.assertEqual(row["kind"], "matrix_norm")
            self.assertEqual(row["regime"], "throughput")
            self.assertIn("#2636", row["issues"])
            n = row["params"]["n"]
            control = f"matrix_norm_frobenius_f64_n{n}"
            if row["params"]["ord"] == "frobenius":
                self.assertNotIn("reference_for", row)
            else:
                self.assertEqual(row["reference_for"], control)
        reduce_rows = gen.gpu_wide_reduction_cases(("nvidia-gpu",))
        self.assertEqual(
            {row["id"]: row["arm"] for row in reduce_rows},
            {
                "reduce_sum_axis0_f64_rank10_wide": "wide",
                "reduce_sum_axis0_f64_rank10_merged-narrow": "merged-narrow",
            },
        )
        for row in reduce_rows:
            self.assertEqual(row["kind"], "wide_reduction")
            self.assertEqual(row["regime"], "throughput")
            self.assertEqual(row["size_elements"], 4**10)
            self.assertIn("#2637", row["issues"])
        self.assertEqual(
            {row["arm"]: row.get("reference_for") for row in reduce_rows},
            {"wide": None, "merged-narrow": "reduce_sum_axis0_f64_rank10_wide"},
        )
        cases = entries("data/instances/gpu_perf_issues.json")
        for issue, generated in (
            ("#2636", [row["id"] for row in norm_rows]),
            ("#2637", [row["id"] for row in reduce_rows]),
        ):
            self.assertEqual(
                {case_id for case_id, case in cases.items() if issue in case["issues"]},
                set(generated),
            )
