"""A withdrawn cell renders as `withdrawn` with its reason, never as its recorded number.

Pins tenferro-benchmark#517. The 2026-11-24 zen5-cpu 4T run recorded the eager
256x256 `grad_sum_svd_s_vjp` / `grad_sum_eigh_vjp` cells with a harness that timed
a second forward pass, and the pages read them as a ~1.95x slowdown. A single-pass
re-measurement does not reproduce it, so `benchmarks/withdrawn-cells.yaml` withdraws
them. The run record stays as collected; the generators apply the registry, and the
dashboard shows a `withdrawn` badge, not a failure.
"""

from __future__ import annotations

import csv
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import format_cpu_ops_results  # noqa: E402
import format_linalg_ad_results  # noqa: E402
import generate_dashboard  # noqa: E402
import result_status  # noqa: E402
import withdrawn_cells  # noqa: E402

RUN_4T = ROOT / "data/results/zen5-cpu/cpu/einsum/20261124_131021"
CSV_4T = RUN_4T / "cpu_ops_t4_20261124_131021.csv"
WITHDRAWN = {("grad_sum_svd_s_vjp", "256x256"), ("grad_sum_eigh_vjp", "256x256")}


class RegistryTest(unittest.TestCase):
    def test_the_two_517_cells_are_withdrawn_and_nothing_else(self) -> None:
        rows = withdrawn_cells.read_rows(CSV_4T)
        withdrawn = {(r["benchmark"], r["shape"], r["backend"]) for r in rows
                     if r["status"] == result_status.WITHDRAWN_STATUS}
        self.assertEqual(withdrawn, {(b, s, "tenferro-eager") for b, s in WITHDRAWN})
        for row in rows:
            if row["status"] == result_status.WITHDRAWN_STATUS:
                self.assertEqual((row["median_ms"], row["iqr_ms"]), ("", ""))
                self.assertIn("second forward", row["reason"])

    def test_other_threads_and_backends_keep_their_numbers(self) -> None:
        with CSV_4T.open(newline="") as f:
            original = list(csv.DictReader(f))
        for before, after in zip(original, withdrawn_cells.read_rows(CSV_4T)):
            if after["status"] != result_status.WITHDRAWN_STATUS:
                self.assertEqual(before["median_ms"], after["median_ms"])

    def test_an_entry_without_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            registry = Path(work) / "benchmarks" / "withdrawn-cells.yaml"
            registry.parent.mkdir()
            registry.write_text(textwrap.dedent("""\
                cells:
                  - run: zen5-cpu/cpu/einsum/20261124_131021
                    benchmark: grad_sum_svd_s_vjp
                    dtype: f64
                    threads: 4
                    shape: 256x256
                    backend: tenferro-eager
                    ref: x
                    reason: y
            """))
            with self.assertRaisesRegex(ValueError, "lacks evidence"):
                withdrawn_cells._entries(registry)

    def test_the_registry_loads_with_its_evidence_present(self) -> None:
        entries = withdrawn_cells._entries()
        self.assertTrue(entries)


class PageTest(unittest.TestCase):
    def test_linalg_page_prints_withdrawn_and_the_reason(self) -> None:
        table = format_linalg_ad_results.format_table(CSV_4T)
        self.assertIn("| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | withdrawn |", table)
        self.assertNotIn("36.898", table)
        self.assertIn("## Withdrawn cells", table)

    def test_cpu_ops_page_prints_withdrawn_too(self) -> None:
        table = format_cpu_ops_results.format_table([CSV_4T])
        self.assertIn("`grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | withdrawn |", table)
        self.assertIn("## Withdrawn cells", table)

    def test_summary_row_is_withdrawn_unverified_with_reason(self) -> None:
        summary = format_linalg_ad_results.build_summary(CSV_4T, {})
        rows = [r for r in summary["rows"] if r["status"] == result_status.WITHDRAWN_STATUS]
        self.assertEqual({(r["case"], r["group"].rsplit("=", 1)[1]) for r in rows}, WITHDRAWN)
        for row in rows:
            self.assertIsNone(row["median"])
            self.assertFalse(row["verified"])
            self.assertTrue(row["reason"])

    def test_no_page_without_a_registry_entry_changes(self) -> None:
        with mock.patch.object(withdrawn_cells, "_entries", return_value={}):
            table = format_linalg_ad_results.format_table(CSV_4T)
        self.assertNotIn("withdrawn", table)
        self.assertIn("36.898 ± 0.056", table)


class DashboardTest(unittest.TestCase):
    ROW = {"case": "grad_sum_svd_s_vjp", "backend": "tenferro-eager", "median": None,
           "status": "withdrawn", "verified": False, "reason": "timed a second forward"}

    def test_withdrawn_is_neither_a_failure_nor_an_anomaly(self) -> None:
        self.assertEqual(generate_dashboard.non_ok_label(self.ROW), "withdrawn")
        self.assertFalse(generate_dashboard.is_tenferro_anomaly(self.ROW))
        self.assertNotIn("status", generate_dashboard.compute_case_flags([self.ROW]))

    def test_withdrawn_is_not_counted_as_a_failed_case(self) -> None:
        outcomes = result_status.case_outcomes([self.ROW])
        self.assertEqual(outcomes["failed"]["count"], 0)

    def test_a_withdrawn_row_without_reason_is_still_an_anomaly(self) -> None:
        self.assertTrue(generate_dashboard.is_tenferro_anomaly({**self.ROW, "reason": ""}))


if __name__ == "__main__":
    unittest.main()
