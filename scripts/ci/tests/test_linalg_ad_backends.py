"""The linalg JVP/VJP table keeps every backend it is given.

`cpu/linalg_jvp_vjp` is derived from the CPU-ops CSV, and the JVP/VJP cases are
registered on `tenferro-trace` only (`push_linalg_jvp_vjp_trace_benches` in
`src/bin/publication_gate.rs`), so today's pages carry no `tenferro-eager` column.
That must be a property of the *measurement*, not of the formatter: a formatter
that drops an unknown backend turns a future eager measurement into a silent
absence, which is the failure mode `docs/cpu-suite-arms.md` exists to prevent.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import format_linalg_ad_results as fmt  # noqa: E402

HEADER = [
    "suite",
    "benchmark",
    "dtype",
    "threads",
    "shape",
    "backend",
    "median_ms",
    "iqr_ms",
    "status",
]


def write_csv(rows) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    )
    writer = csv.writer(handle)
    writer.writerow(HEADER)
    writer.writerows(rows)
    handle.close()
    return Path(handle.name)


def row(backend: str, value: str, benchmark: str = "grad_sum_qr_jvp") -> list[str]:
    return ["large", benchmark, "f64", "8", "1024x1024", backend, value, "0.1", "ok"]


class LinalgAdBackendColumns(unittest.TestCase):
    def test_trace_only_page_has_no_eager_column(self):
        path = write_csv([row("tenferro-trace", "11.0"), row("pytorch-cpu", "2.0")])
        table = fmt.format_table(path)
        self.assertIn("tenferro-rs trace mode (ms)", table)
        self.assertNotIn("tenferro-rs eager mode (ms)", table)

    def test_eager_rows_are_rendered_not_dropped(self):
        path = write_csv(
            [
                row("tenferro-eager", "3.0"),
                row("tenferro-trace", "11.0"),
                row("pytorch-cpu", "2.0"),
            ]
        )
        table = fmt.format_table(path)
        self.assertIn("tenferro-rs eager mode (ms)", table)
        self.assertIn("| 3.000 ± 0.100 |", table)

    def test_an_unknown_backend_keeps_its_column(self):
        path = write_csv([row("tenferro-trace", "11.0"), row("some-new-ref", "4.0")])
        table = fmt.format_table(path)
        self.assertIn("some-new-ref", table)

    def test_declared_order_is_kept_then_extras_sorted(self):
        self.assertEqual(
            fmt.backend_order({"jax-cpu", "tenferro-trace", "zeta-ref", "alpha-ref"}),
            ["tenferro-trace", "jax-cpu", "alpha-ref", "zeta-ref"],
        )


if __name__ == "__main__":
    unittest.main()
