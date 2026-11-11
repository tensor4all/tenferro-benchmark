"""The arms matrix in `docs/cpu-suite-arms.md` follows the committed summaries.

A re-ingestion can add an arm without touching that page, and the page then says
a column does not exist while the published report shows it. That is exactly what
happened when amd-cpu and mac-cpu were republished: `cpu/linalg_jvp_vjp` gained
its eager column and the matrix still read "0 eager ... awaiting the next
collection". The numbers below are facts of the committed summaries, so the test
reads them instead of trusting the prose.
"""

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("zen5-cpu", "amd-cpu", "mac-cpu")


def summary(suite: str, profile: str) -> dict:
    path = ROOT / "result" / profile / "cpu" / f"{suite}.summary.json"
    return json.loads(path.read_text())


def rows_by_backend(suite: str, profile: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in summary(suite, profile)["rows"]:
        counts[row["backend"]] = counts.get(row["backend"], 0) + 1
    return counts


def matrix_line(doc: str, suite: str) -> str:
    for line in doc.splitlines():
        if line.startswith(f"| `{suite}`"):
            return line
    raise AssertionError(f"docs/cpu-suite-arms.md: no matrix row for {suite}")


class ArmsMatrixTest(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = (ROOT / "docs/cpu-suite-arms.md").read_text()

    def test_cpu_ops_eager_row_counts(self):
        line = matrix_line(self.doc, "cpu/cpu_ops")
        match = re.search(r"zen5 (\d+) / amd (\d+) / mac (\d+)", line)
        self.assertIsNotNone(match, line)
        expected = [rows_by_backend("cpu_ops", p).get("tenferro-eager", 0) for p in PROFILES]
        self.assertEqual([int(g) for g in match.groups()], expected, line)

    def test_linalg_jvp_vjp_eager_and_trace_row_counts(self):
        line = matrix_line(self.doc, "cpu/linalg_jvp_vjp")
        match = re.search(
            r"zen5 \*\*(\d+)\*\* eager / (\d+) trace; amd and mac \*\*(\d+) / (\d+)\*\*", line
        )
        self.assertIsNotNone(match, line)
        zen5 = rows_by_backend("linalg_jvp_vjp", "zen5-cpu")
        amd = rows_by_backend("linalg_jvp_vjp", "amd-cpu")
        mac = rows_by_backend("linalg_jvp_vjp", "mac-cpu")
        self.assertEqual(int(match.group(1)), zen5.get("tenferro-eager", 0), line)
        self.assertEqual(int(match.group(2)), zen5.get("tenferro-trace", 0), line)
        # The row claims amd and mac agree; if they ever differ the prose has to
        # stop reading "amd and mac".
        self.assertEqual(amd.get("tenferro-eager", 0), mac.get("tenferro-eager", 0), line)
        self.assertEqual(amd.get("tenferro-trace", 0), mac.get("tenferro-trace", 0), line)
        self.assertEqual(int(match.group(3)), amd.get("tenferro-eager", 0), line)
        self.assertEqual(int(match.group(4)), amd.get("tenferro-trace", 0), line)

    def test_no_page_claims_a_column_is_missing_when_it_is_published(self):
        # The drift this test exists for: the page said the summaries carry no
        # eager JVP/VJP rows while two of them did.
        self.assertNotIn("still show **0 eager JVP/VJP rows**", self.doc)


if __name__ == "__main__":
    unittest.main()
