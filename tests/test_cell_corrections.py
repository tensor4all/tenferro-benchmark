"""A corrected cell prints its re-measurement; a withdrawn cell prints no number.

Pins tenferro-benchmark#517. The 2026-11-24 zen5-cpu 4T run recorded the eager
256x256 `grad_sum_svd_s_vjp` / `grad_sum_eigh_vjp` cells with a harness that timed
a second forward pass, and the pages read them as a ~1.95x slowdown. A single-pass
re-measurement of those two cells is recorded under `data/cell-corrections/`:
`svd_s_vjp` is not distinguishable from 9a850b06 and is withdrawn, `eigh_vjp` is
resolvable and prints its corrected number. The run record stays as collected; the
generators apply `benchmarks/cell-corrections.yaml`, and the dashboard shows a
`withdrawn` badge, not a failure.
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

import cell_corrections  # noqa: E402
import format_cpu_ops_results  # noqa: E402
import format_linalg_ad_results  # noqa: E402
import generate_dashboard  # noqa: E402
import result_status  # noqa: E402

RUN_4T = ROOT / "data/results/zen5-cpu/cpu/einsum/20261124_131021"
CSV_4T = RUN_4T / "cpu_ops_t4_20261124_131021.csv"


def applied(rows):
    return {(r["benchmark"], r["shape"], r["backend"], r["correction_action"])
            for r in rows if r.get("correction_action")}


class RegistryTest(unittest.TestCase):
    def test_exactly_the_two_517_cells_are_changed(self) -> None:
        rows = cell_corrections.read_rows(CSV_4T)
        self.assertEqual(applied(rows), {
            ("grad_sum_svd_s_vjp", "256x256", "tenferro-eager", "withdraw"),
            ("grad_sum_eigh_vjp", "256x256", "tenferro-eager", "correct"),
        })
        by_name = {r["benchmark"]: r for r in rows if r.get("correction_action")}
        self.assertEqual((by_name["grad_sum_svd_s_vjp"]["median_ms"],
                          by_name["grad_sum_svd_s_vjp"]["status"]), ("", "withdrawn"))
        self.assertEqual((by_name["grad_sum_eigh_vjp"]["median_ms"],
                          by_name["grad_sum_eigh_vjp"]["status"]), ("10.228000", "ok"))

    def test_every_other_row_keeps_its_recorded_number(self) -> None:
        with CSV_4T.open(newline="") as f:
            original = list(csv.DictReader(f))
        for before, after in zip(original, cell_corrections.read_rows(CSV_4T)):
            if not after.get("correction_action"):
                self.assertEqual(before, after)

    def _registry(self, body: str) -> Path:
        work = Path(tempfile.mkdtemp())
        (work / "benchmarks").mkdir()
        (work / "evidence").mkdir()
        registry = work / "benchmarks" / "cell-corrections.yaml"
        registry.write_text(textwrap.dedent(body))
        return registry

    def test_an_entry_without_evidence_is_rejected(self) -> None:
        registry = self._registry("""\
            cells:
              - {run: a/b/c/d, benchmark: x, dtype: f64, threads: 4, shape: 2x2,
                 backend: tenferro-eager, action: withdraw, ref: r, reason: y}
        """)
        with self.assertRaisesRegex(ValueError, "lacks evidence"):
            cell_corrections._entries(registry)

    def test_a_correction_without_its_number_is_rejected(self) -> None:
        registry = self._registry("""\
            cells:
              - {run: a/b/c/d, benchmark: x, dtype: f64, threads: 4, shape: 2x2,
                 backend: tenferro-eager, action: correct, ref: r, reason: y, evidence: evidence}
        """)
        with self.assertRaisesRegex(ValueError, "lacks median_ms, iqr_ms"):
            cell_corrections._entries(registry)


class PageTest(unittest.TestCase):
    def test_linalg_page_withdraws_svd_and_corrects_eigh(self) -> None:
        table = format_linalg_ad_results.format_table(CSV_4T)
        self.assertIn("| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | withdrawn |", table)
        self.assertIn("| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 10.228 ± 0.085 (corrected) |",
                      table)
        table_rows = [line for line in table.splitlines() if line.startswith("| large |")]
        self.assertFalse([line for line in table_rows if "36.898" in line or "20.050 ±" in line])
        self.assertIn("## Corrected and withdrawn cells", table)

    def test_cpu_ops_page_shows_the_same_two_cells(self) -> None:
        table = format_cpu_ops_results.format_table([CSV_4T])
        self.assertIn("`grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | withdrawn |", table)
        self.assertIn("`grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 10.228 ± 0.085 (corrected) |", table)

    def test_summary_rows_carry_the_reason(self) -> None:
        summary = format_linalg_ad_results.build_summary(CSV_4T, {})
        rows = {r["case"]: r for r in summary["rows"]
                if r["backend"] == "tenferro-eager" and r["group"].endswith("threads=4 / shape=256x256")
                and r["case"] in ("grad_sum_svd_s_vjp", "grad_sum_eigh_vjp")}
        self.assertEqual((rows["grad_sum_svd_s_vjp"]["status"], rows["grad_sum_svd_s_vjp"]["median"],
                          rows["grad_sum_svd_s_vjp"]["verified"]), ("withdrawn", None, False))
        self.assertEqual((rows["grad_sum_eigh_vjp"]["status"], rows["grad_sum_eigh_vjp"]["median"]),
                         ("ok", 10.228))
        for row in rows.values():
            self.assertIn("second forward", row["reason"])

    def test_without_registry_entries_the_page_is_the_run_record(self) -> None:
        with mock.patch.object(cell_corrections, "_entries", return_value={}):
            table = format_linalg_ad_results.format_table(CSV_4T)
        self.assertNotIn("withdrawn", table)
        self.assertNotIn("corrected", table)
        self.assertIn("36.898 ± 0.056", table)


class DashboardTest(unittest.TestCase):
    ROW = {"case": "grad_sum_svd_s_vjp", "backend": "tenferro-eager", "median": None,
           "status": "withdrawn", "verified": False, "reason": "timed a second forward"}

    def test_withdrawn_is_neither_a_failure_nor_an_anomaly(self) -> None:
        self.assertEqual(generate_dashboard.non_ok_label(self.ROW), "withdrawn")
        self.assertFalse(generate_dashboard.is_tenferro_anomaly(self.ROW))
        self.assertNotIn("status", generate_dashboard.compute_case_flags([self.ROW]))

    def test_withdrawn_is_not_counted_as_a_failed_case(self) -> None:
        self.assertEqual(result_status.case_outcomes([self.ROW])["failed"]["count"], 0)

    def test_a_withdrawn_row_without_reason_is_still_an_anomaly(self) -> None:
        self.assertTrue(generate_dashboard.is_tenferro_anomaly({**self.ROW, "reason": ""}))


if __name__ == "__main__":
    unittest.main()
